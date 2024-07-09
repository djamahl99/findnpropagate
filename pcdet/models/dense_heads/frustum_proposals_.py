import copy
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

from pcdet.utils.box_utils import boxes_to_corners_3d
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from torchvision.ops import box_iou, batched_nms, nms

from ..model_utils import model_nms_utils

from ..preprocessed_detector import PreprocessedDetector

def get_cam_frustum(xyzxyz):
    whl = xyzxyz[3:] - xyzxyz[0:3]
    center = (xyzxyz[3:] + xyzxyz[0:3]) / 2

    template = xyzxyz.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners = whl[None, :].repeat(8, 1) * template[:, :]
    corners[:, 0:3] += center

    return corners

class FrustumProposerOG(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
        lq=0.32, uq=1.0, iou_weight=1.0, dist_weight=0.5, density_weight=0.5, \
                 min_cam_iou=0.2, size_min=0.6, size_max=1.0, ry_min=0.0, ry_max=torch.pi, centreq=0.5, num_mags=10
    ):
        super(FrustumProposerOG, self).__init__()

        self.image_order = [2, 0, 1, 5, 3, 4]
        # self.image_size = [512, 800]
        self.image_size = [900, 1600]

        self.num_sample_pts = 100

        self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

        self.first_stage_topk = 50
        self.second_stage_topk = 1

        self.mags_min = 0.0
        self.mags_max = 1.0

        self.ry_min = ry_min
        self.ry_max = ry_max

        self.size_min = size_min
        self.size_max = size_max

        # how many of each box variation we will include
        self.num_mags = num_mags
        self.num_sizes = 5
        self.num_rotations = 20

        self.lq = lq
        self.uq = uq
        self.centreq = centreq

        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.density_weight = density_weight
        self.min_cam_iou = min_cam_iou

        num_proposals = self.num_mags * self.num_sizes * self.num_rotations

        self.frustum_min = torch.tensor(1.0, device='cuda')
        self.frustum_max = torch.tensor(60.0, device='cuda')

        print(f'will be generating a total of {num_proposals} in each frustum')

        self.x_size = 50
        self.y_size = 50

        # preds_path = '/home/uqdetche/OpenPCDet/tools/coco_val_'
        preds_path = '/home/uqdetche/GLIP/OWL_'
        camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.image_detector = PreprocessedDetector([preds_path + f"{cam_name}.json" for cam_name in camera_names])

        anchors = [
            [4.63, 1.97, 1.74],
            [6.93, 2.51, 2.84], 
            [6.37, 2.85, 3.19],
            [10.5, 2.94, 3.47], 
            [12.29, 2.90, 3.87],
            [0.50, 2.53, 0.98],
            [2.11, 0.77, 1.47],
            [1.70, 0.60, 1.28],
            [0.73, 0.67, 1.77],
            [0.41, 0.41, 1.07]
        ]
        anchors = torch.tensor(anchors, dtype=torch.float32, requires_grad=False, device='cuda')
        self.anchors = anchors
        size_variations = torch.linspace(0.6, 1.0, steps=self.num_sizes, device='cuda')
        base_rotations = torch.linspace(0, torch.pi, steps=self.num_rotations, device='cuda')
        self.base_boxes = torch.zeros((anchors.shape[0], self.num_rotations, self.num_sizes, 7), device='cuda')

        for i in range(anchors.shape[0]):
            self.base_boxes[i, :, :, [3, 4, 5]] = anchors[i]

        for i in range(self.num_rotations):
            self.base_boxes[:, i, :, -1] = base_rotations[i]

        # it = 0
        # for i in range(base_coord.shape[0]):
        #     for j in range(3):
        #         self.base_boxes[:, :, it, :, j] = base_coord[i]
        #         it += 1

        for i, m in enumerate(size_variations):
            self.base_boxes[:, :, i, [3, 4, 5]] = self.base_boxes[:, :, i, [3, 4, 5]] * m

        # print('base boxes', self.base_boxes)

        self.base_corners = boxes_to_corners_3d(self.base_boxes.reshape(-1, 7)).reshape(self.anchors.shape[0], -1, 8, 3)
        self.base_boxes = self.base_boxes.reshape(self.anchors.shape[0], -1, 7)
        
        self.bev_pos = self.create_2D_grid(self.x_size, self.y_size)

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

        pc_bev_pos = self.bev_pos.clone().reshape(-1, 2).cuda()
        pc_bev_pos[:, 0] = (pc_bev_pos[:, 0] - pc_bev_pos[:, 0].min()) / (pc_bev_pos[:, 0].max() - pc_bev_pos[:, 0].min()) * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
        pc_bev_pos[:, 1] = (pc_bev_pos[:, 1] - pc_bev_pos[:, 1].min()) / (pc_bev_pos[:, 1].max() - pc_bev_pos[:, 1].min()) * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
        
        self.pc_bev_pos = pc_bev_pos
        self.bev_pos_long = (self.bev_pos.clone().reshape(-1, 2) - 0.5).long()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base
    
    def get_frustum_bev_mask(self, frust_box: torch.tensor):
        frust_bev_box = torch.cat([frust_box[[2*i, 2*i+1], :].mean(dim=0)[None, :] for i in range(4)], dim=0)

        frust_center_far = frust_bev_box[2:, :2].mean(dim=0)

        frust_left_vector = frust_bev_box[2, :2]
        frust_right_vector = frust_bev_box[3, :2]
        frust_center_vector = frust_center_far #- frust_center_close

        frust_left_vector_norm = frust_left_vector / torch.norm(frust_left_vector, keepdim=True)
        frust_right_vector_norm = frust_right_vector / torch.norm(frust_right_vector, keepdim=True)
        frust_center_vector_norm = frust_center_vector / torch.norm(frust_center_vector, keepdim=True)

        bev_pos = self.pc_bev_pos.clone().reshape(1, -1, 2)
        bev_mags = torch.norm(bev_pos, keepdim=True, dim=-1)
        bev_pos_norm = bev_pos / bev_mags

        left_to_bev = bev_pos_norm.clone() - frust_left_vector_norm.reshape(1, 1, 2)
        # left_to_bev = bev_pos.clone() - frust_left_vector.reshape(1, 1, 2)
        left_to_bev = left_to_bev / torch.norm(left_to_bev, keepdim=True, dim=-1)
        right_to_bev = bev_pos_norm.clone() - frust_right_vector_norm.reshape(1, 1, 2)
        # right_to_bev = bev_pos.clone() - frust_right_vector.reshape(1, 1, 2)
        right_to_bev = right_to_bev / torch.norm(right_to_bev, keepdim=True, dim=-1)

        left_to_bev = left_to_bev.reshape(-1, 2)
        right_to_bev = right_to_bev.reshape(-1, 2)

        cos_dir_lr = left_to_bev * right_to_bev
        cos_dir_lr = cos_dir_lr.sum(dim=-1)
        cos_dir_center = frust_center_vector_norm.reshape(-1, 1, 2) * bev_pos_norm
        cos_dir_center = cos_dir_center.sum(dim=-1)
        cos_dir_center = cos_dir_center.reshape(-1)

        frust_min_mag = frust_bev_box.norm(dim=-1).min()
        frust_max_mag = frust_bev_box.norm(dim=-1).max()

        bev_mags = bev_mags.reshape(-1)

        # bev points in frustum
        dirbools = torch.bitwise_and((cos_dir_lr < 0), (cos_dir_center > 0))
        magbools = torch.bitwise_and((bev_mags >= frust_min_mag), (bev_mags < frust_max_mag))

        return torch.bitwise_and(dirbools, magbools)

    def frustum_bev_nms(self, batch_bev_masks, batch_scores, batch_labels, nms_thresh=0.5):
        N = batch_bev_masks.shape[0]
        batch_indices = torch.arange(0, N)
        keep = torch.ones_like(batch_indices, dtype=torch.bool)
        order = torch.argsort(- batch_scores)

        combined = {x: [] for x in range(N)}

        for i in range(N):
            if keep[i]:
                curr_idx = order[i]
                cur_bev_mask = batch_bev_masks[curr_idx]

                for j in range(i + 1, N):
                    other_idx = order[j]
                    other_mask = batch_bev_masks[other_idx]

                    intersection = torch.bitwise_and(cur_bev_mask, other_mask).sum()
                    union = torch.bitwise_or(cur_bev_mask, other_mask).sum()

                    bev_iou = intersection / (union + 1e-8)

                    if bev_iou > nms_thresh and batch_labels[curr_idx] == batch_labels[other_idx]:
                        keep[j] = 0

                        combined[curr_idx.item()].append(other_idx.item())
                        combined[curr_idx.item()].extend(combined[other_idx.item()])


        combined = {i: list(set(combined[i])) for i in combined.keys()}

        return order[keep], combined

    def get_proposals(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        # images = batch_dict['camera_imgs']

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        frustum_min = torch.tensor(1.0, device='cuda')
        frustum_max = torch.tensor(60.0, device='cuda')

        
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        proposal_boxes = []
        frust_labels = []
        frust_batch_idx = []
        frust_scores = []

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            batch_frusts = []
            batch_cam_boxes = []
            batch_frusts_cams = []
            batch_frusts_points = []
            batch_frusts_mins = []
            batch_frusts_maxs = []
            batch_weighted_centres = []
            batch_bev_masks = []
            batch_scores = []
            batch_labels = []

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]
                # cam_points = proj_points[c, proj_points_cam_mask[c]]

                cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                cam_points = cam_points[cam_mask]

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < 0.1:
                        continue

                    x1, y1, x2, y2 = box.cpu()

                    # if (x2 - x1) * (y2 - y1) > (1600 * 900) *0.5:
                        # continue

                    on_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    box_points = cam_points[on_box]

                    if box_points.numel() > 0:
                        cur_frustum_min = torch.quantile(box_points[:, 2], self.lq)
                        cur_frustum_max = torch.quantile(box_points[:, 2], self.uq)

                        box_centre = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2]).reshape(1, -1).cuda()
                        dists = torch.cdist(box_centre, box_points[:, 0:2])
                        dists_ranking = torch.softmax(- dists, dim=-1) # softmin

                        weighted_centre_cam = dists_ranking @ box_points
                        # weighted_centre_cam = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2, cur_frustum_min[None].cpu()]).reshape(1, -1).cuda()
                        # cur_centre_z = torch.quantile(box_points[:, 2], self.centreq)
                        # weighted_centre_cam = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2, cur_centre_z[None].cpu()]).reshape(1, -1).cuda()

                        weighted_centre_xyz = self.get_geometry_at_image_coords(weighted_centre_cam.reshape(-1, 3), [c], [b], 
                            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                        )
                        # weighted_centre_cam = torch.quantile(box_points, q=0.5, dim=0)
                    else:
                        # print('no pts in box')
                        cur_frustum_min = self.frustum_min
                        cur_frustum_max = self.frustum_max
                        weighted_centre_xyz = None

                    box = box.cuda()
                    xyzxyz = torch.cat([box[0][None], box[1][None], cur_frustum_min[None], box[2][None], box[3][None], cur_frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    bev_mask = self.get_frustum_bev_mask(frust_box)

                    batch_frusts.append(frust_box)
                    batch_cam_boxes.append(box)
                    batch_frusts_cams.append(c)
                    batch_frusts_points.append(box_points)
                    batch_weighted_centres.append(weighted_centre_xyz)
                    batch_bev_masks.append(bev_mask[None])
                    batch_scores.append(score[None])
                    batch_labels.append(label[None])

            if len(batch_bev_masks) == 0:
                proposal_boxes = torch.zeros((0, 7), device='cuda')
                frust_labels = torch.zeros((0), dtype=torch.long)
                frust_scores = torch.zeros((0))
                frust_batch_idx = torch.zeros((0), dtype=torch.long)

                return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

            # do frustum NMS to combine frustums that align well (as objects can appear on multiple cameras)
            batch_bev_masks = torch.cat(batch_bev_masks, dim=0)
            batch_scores = torch.cat(batch_scores, dim=0)

            # frustum nms
            indices, combined = self.frustum_bev_nms(batch_bev_masks, batch_scores, batch_labels, nms_thresh=0.4)
            # print('remaining', indices.shape[0], 'removed', (batch_bev_masks.shape[0] - indices.shape[0]))

            # combine kept frustums with the ones that were removed
            for valid_idx in indices:
                valid_idx = valid_idx.item()

                if len(combined[valid_idx]) == 0:
                    continue

                base_frust = batch_frusts[valid_idx].cpu()
                base_centre = base_frust.mean(dim=0)
                base_dists = torch.cdist(base_frust, base_centre.reshape(1, 3)).reshape(-1)

                for comb_idx in combined[valid_idx]:
                    comb_frust = batch_frusts[comb_idx].cpu()
                    comb_dists = torch.cdist(base_frust, base_centre.reshape(1, 3)).reshape(-1)

                    # replace with corners that are further away, enlarging the frustum
                    further_dists = comb_dists > base_dists
                    base_frust[further_dists] = comb_frust[further_dists]

                batch_frusts[valid_idx] = base_frust.to(batch_frusts[valid_idx].device)

            for i, (frust_box, c, box, box_points, weighted_centre_xyz, label) in \
                    enumerate(zip(batch_frusts, batch_frusts_cams, batch_cam_boxes, \
                                                        batch_frusts_points, batch_weighted_centres, batch_labels)):
                if i not in indices:
                    continue

                frust_bev_box = torch.cat([frust_box[[2*i, 2*i+1], :].mean(dim=0)[None, :] for i in range(4)], dim=0)

                # instead use 100 evenly spaced at the center of the frustum
                mags = torch.linspace(self.mags_min, self.mags_max, self.num_mags, device='cuda').reshape(-1, 1).repeat(1, 3)

                # need 3d one
                frust_center_close = frust_bev_box[:2, :].mean(dim=0)
                frust_center_far = frust_bev_box[2:, :].mean(dim=0)

                center_vec = (frust_center_far - frust_center_close)
                bev_pts_xyz = frust_center_close.reshape(1, 3) + center_vec * mags

                curr_corner_proposals = self.base_corners[label.item()].clone()
                curr_corner_proposals = curr_corner_proposals.unsqueeze(0)
                curr_corner_proposals = curr_corner_proposals + bev_pts_xyz[:, None, None, :]
                
                # box version of the corners -> should be [N, 7]
                curr_box_proposals = self.base_boxes[label.item()].clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1) # [M, N, 7] M = bev_pts_xyz.shape[0]
                curr_box_proposals[..., 0:3] = curr_box_proposals[..., 0:3] + bev_pts_xyz[:, None, :]
                
                curr_corner_proposals = curr_corner_proposals.reshape(-1, 8, 3)
                curr_box_proposals = curr_box_proposals.reshape(-1, 7)

                # now move back so the front of the box is now where the centre was
                closest_ranking = torch.softmax(- curr_corner_proposals.clone().norm(dim=2), dim=1) # softmin
                weighted_front_centres = (closest_ranking.reshape(-1, 8, 1) * curr_corner_proposals).sum(dim=1)
                
                front_to_centre = (curr_box_proposals[..., 0:3] - weighted_front_centres)
                curr_box_proposals[..., 0:3] += front_to_centre
                curr_corner_proposals += front_to_centre.reshape(-1, 1, 3)

                image_pos, _ = self.project_to_camera(batch_dict, curr_corner_proposals.reshape(-1, 3), b, c)
                image_pos = image_pos[..., :2].reshape(-1, 8, 2)

                image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
                image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

                x1, y1, x2, y2 = image_pos[..., 0].min(dim=-1).values, image_pos[..., 1].min(dim=-1).values, image_pos[..., 0].max(dim=-1).values, image_pos[..., 1].max(dim=-1).values

                proj_boxes = torch.zeros((x1.shape[0], 4))
                proj_boxes[:, 0] = x1
                proj_boxes[:, 1] = y1
                proj_boxes[:, 2] = x2
                proj_boxes[:, 3] = y2

                ious = box_iou(proj_boxes, box.cpu().reshape(1, -1)).reshape(-1)

                ious_ranked = (ious.clone() - ious.min()) / (ious.max() - ious.min() + 1e-8)

                if weighted_centre_xyz is not None:
                    dists = torch.cdist(weighted_front_centres.reshape(-1, 3), weighted_centre_xyz.reshape(1, 3)).reshape(-1)
                    # dists_ranked = torch.softmax(-dists, dim=0)
                    dists_ranked = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
                    dists_ranked = 1 - dists_ranked

                    first_stage_scores = ious_ranked * self.iou_weight + dists_ranked.cpu() * self.dist_weight
                else:
                    first_stage_scores = ious_ranked * self.iou_weight


                valid_proposals = torch.topk(first_stage_scores, k=self.first_stage_topk).indices
                curr_corner_proposals = curr_corner_proposals[valid_proposals]
                curr_box_proposals = curr_box_proposals[valid_proposals]
                ious = ious[valid_proposals]
                first_stage_scores = first_stage_scores[valid_proposals]
                image_pos = image_pos[valid_proposals]
                proj_boxes = proj_boxes[valid_proposals]

                # calculate point density
                box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                    camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                    post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                )
                
                num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])

                if box_points_xyz.shape[0] > 0:
                    point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(box_points_xyz.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
                    point_box_indices = point_box_indices.reshape(-1)
                    
                    for i in range(curr_box_proposals.shape[0]):
                        num_pts_in_boxes[i] = (point_box_indices == i).sum()


                soft_densities = (num_pts_in_boxes - num_pts_in_boxes.min()) / (num_pts_in_boxes.max() - num_pts_in_boxes.min() + 1e-8)
                second_stage_scores = soft_densities * self.density_weight + first_stage_scores

                valid_proposals = torch.topk(second_stage_scores, k=self.second_stage_topk).indices
                curr_corner_proposals = curr_corner_proposals[valid_proposals]
                curr_box_proposals = curr_box_proposals[valid_proposals]
                ious = ious[valid_proposals]
                second_stage_scores = second_stage_scores[valid_proposals]
                image_pos = image_pos[valid_proposals]
                proj_boxes = proj_boxes[valid_proposals]

                if ious.max() < self.min_cam_iou:
                    # print('ignored, max iou=', ious.max())
                    continue

                proposal_boxes.append(curr_box_proposals[None])
                frust_batch_idx.append(b)
                frust_labels.append(label)
                frust_scores.append(score)

        if len(proposal_boxes) > 0:
            proposal_boxes = torch.cat(proposal_boxes, dim=0).reshape(-1, 7)
            frust_labels = torch.tensor(frust_labels)
            frust_scores = torch.tensor(frust_scores)
            frust_batch_idx = torch.tensor(frust_batch_idx)
        else:
            proposal_boxes = torch.zeros((0, 7), device='cuda')
            frust_labels = torch.zeros((0), dtype=torch.long)
            frust_scores = torch.zeros((0))
            frust_batch_idx = torch.zeros((0), dtype=torch.long)

        print('boxes, labels, idx', proposal_boxes.shape, frust_labels.shape, frust_batch_idx.shape)
        print('boxes, labels, idx', proposal_boxes.device, frust_labels.device, frust_batch_idx.device)

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        batch_size = batch_dict['batch_size']

        cur_coords = points.clone()

        # print('current coords', cur_coords)
        # print('image_paths', batch_dict['image_paths'])

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        cur_img_aug_matrix = img_aug_matrix[batch_idx, [cam_idx]]
        cur_lidar_aug_matrix = lidar_aug_matrix[batch_idx]
        cur_lidar2image = lidar2image[batch_idx, [cam_idx]]

        # inverse aug
        cur_coords -= cur_lidar_aug_matrix[:3, 3]
        cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
            cur_coords.transpose(1, 0)
        )
        # lidar2image
        cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
        cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
        # get 2d coords
        cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :].clone(), 1e-5, 1e5)
        cur_coords[:, :2, :] /= cur_coords[:, 2:3, :].clone()

        # do image aug
        cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
        cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
        cur_coords = cur_coords[:, :3, :].transpose(1, 2)

        # normalize coords for grid sample
        # cur_coords = cur_coords[..., [1, 0]]

        # filter points outside of images
        on_img = (
            (cur_coords[..., 1] < self.image_size[0])
            & (cur_coords[..., 1] >= 0)
            & (cur_coords[..., 0] < self.image_size[1])
            & (cur_coords[..., 0] >= 0)
        )

        return cur_coords, on_img

    def get_geometry_at_image_coords(self, image_coords, cam_idx, batch_idx, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        # image_coords like (L, 3) # 

        camera2lidar_rots = camera2lidar_rots[batch_idx, cam_idx].to(torch.float)
        camera2lidar_trans = camera2lidar_trans[batch_idx, cam_idx].to(torch.float)
        intrins = intrins[batch_idx, cam_idx].to(torch.float)
        post_rots = post_rots[batch_idx, cam_idx].to(torch.float)
        post_trans = post_trans[batch_idx, cam_idx].to(torch.float)

        # B, N, _ = camera2lidar_trans.shape
        L = image_coords.shape[0]

        # undo post-transformation
        # B x N x L x 3
        points = image_coords - post_trans.view(L, 3)
        points = torch.inverse(post_rots).view(L, 3, 3).matmul(points.unsqueeze(-1)).reshape(L, 3)
        
        # cam_to_lidar
        points = torch.cat((points[:, :2] * points[:, 2:3], points[:, 2:3]), -1)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(L, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += camera2lidar_trans.view(L, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots[batch_idx].view(L, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans[batch_idx].view(L, 3)#.repeat(1, N, 1, 1)

        return points

    def forward(self, batch_dict):
        bboxes = self.get_bboxes(batch_dict)
        batch_dict['final_box_dicts'] = bboxes

        assert not self.training, "not trainable!"
        return batch_dict

    def get_bboxes(self, batch_dict):
        proposed_boxes, proposed_labels, proposed_scores, proposed_batch_idx = self.get_proposals(batch_dict)

        empty_dict = dict(pred_boxes=[], pred_scores=[], pred_labels=[])
        ret_dict = [empty_dict] * batch_dict['batch_size']


        for k in range(batch_dict['batch_size']):
            mask = proposed_batch_idx == k

            # ret_dict.append()
            print('boxes', proposed_boxes[mask].shape)
            print('scores', proposed_scores[mask].shape)
            print('labels', proposed_labels[mask].shape)

            ret_dict[k]['pred_boxes'] = proposed_boxes[mask]
            ret_dict[k]['pred_scores'] = proposed_scores[mask]
            ret_dict[k]['pred_labels'] = proposed_labels[mask].int() + 1

        return ret_dict 
