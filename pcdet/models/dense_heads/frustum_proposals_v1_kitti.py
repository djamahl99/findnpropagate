import copy
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
import torchvision

from pcdet.utils.box_utils import boxes_to_corners_3d
from pcdet.utils.calibration_kitti import CalibrationTorch
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from torchvision.ops import box_iou, batched_nms, nms
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ..model_utils import model_nms_utils
import time

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

class FrustumProposerOGKITTI(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
        lq=0.336, uq=0.356, iou_w=0.95, dst_w=0.226, dns_w=0.05, \
                 min_cam_iou=0.3, size_min=0.957, size_max=1.2, ry_min=0.0, ry_max=torch.pi, cq=0.46, num_mags=6,
                 max_dist=70, num_sizes=4, num_rotations=10, topk=1, nms_2d=0.7, nms_3d=1.0, score_thr=0.1, nms_normal=0.7, clamp_bottom=0
    ):
        super(FrustumProposerOGKITTI, self).__init__()

        self.save_blender = False

        # whether need blender infos
        self.SAVE_BLEND_FILES = model_cfg is not None and model_cfg.get('SAVE_BLEND', False)
        self.MULTICAM_IOU = model_cfg is not None and model_cfg.get('MULTICAM_IOU', False)
        self.OCCL_MULT = model_cfg is not None and model_cfg.get('OCCL_MULT', False)
        self.MULT = model_cfg is not None and model_cfg.get('MULT', False)

        # if self.SAVE_BLEND_FILES:
            # score_thr = 0.3
        
        self.search_depth = None
        self.clamp_bottom = 0
        self.rand_center = False
        self.aln_w = 0
        self.ego_w = 0
        self.occl_w = 0
        if model_cfg is not None and 'PARAMS' in model_cfg:
            params_dict = model_cfg.PARAMS

            lq = params_dict.get('lq', lq)
            uq = params_dict.get('uq', uq)
            iou_w = params_dict.get('iou_w', iou_w)
            dst_w = params_dict.get('dst_w', dst_w)
            dns_w = params_dict.get('dns_w', dns_w)
            self.aln_w = params_dict.get('aln_w', self.aln_w)
            self.ego_w = params_dict.get('ego_w', self.ego_w)
            min_cam_iou = params_dict.get('min_cam_iou', min_cam_iou) 
            size_min = params_dict.get('size_min', size_min)
            size_max = params_dict.get('size_max', size_max)
            cq = params_dict.get('cq', cq) 
            num_mags = params_dict.get('num_mags', num_mags)
            max_dist = params_dict.get('max_dist', max_dist)
            num_sizes = params_dict.get('num_sizes', num_sizes)
            num_rotations = params_dict.get('num_rotations', num_rotations)
            topk = params_dict.get('topk', topk)
            score_thr = params_dict.get('score_thr', score_thr)
            nms_2d = params_dict.get('nms_2d', nms_2d)
            nms_3d = params_dict.get('nms_3d', nms_3d)
            nms_normal = params_dict.get('nms_normal', nms_normal)
            self.clamp_bottom = params_dict.get('clamp_bottom', clamp_bottom)
            self.rand_center = params_dict.get('rand_center', self.rand_center)
            self.occl_w = params_dict.get('occl_w', self.occl_w)
            ry_min = params_dict.get('ry_min', ry_min)
            ry_max = params_dict.get('ry_max', ry_max)

            self.search_depth = params_dict.get('search_depth', None)
            print(params_dict)

        self.nms_normal = nms_normal

        self.image_order = [2, 0, 1, 5, 3, 4]
        # self.image_size = [512, 800]
        self.image_size = [900, 1600]

        # self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.point_cloud_range = [0, -40, -3, 70.4, 40, 1]

        self.topk = topk
        self.nms_3d = nms_3d
        assert self.nms_3d == 0, 'DO NOT USE!'
        self.nms_2d = nms_2d

        self.mags_min = 0.0
        self.mags_max = 1.0

        self.ry_min = ry_min
        self.ry_max = ry_max

        self.size_min = size_min
        self.size_max = size_max

        # how many of each box variation we will include
        self.num_mags = num_mags
        self.num_sizes = num_sizes
        self.num_rotations = num_rotations

        self.max_dist = max_dist
        self.score_thr = score_thr

        self.lq = lq
        self.uq = uq
        self.cq = cq

        self.iou_w = iou_w
        self.dst_w = dst_w
        self.dns_w = dns_w
        self.min_cam_iou = min_cam_iou

        num_proposals = self.num_mags * self.num_sizes * self.num_rotations

        self.frustum_min = torch.tensor(2.0, device='cuda')
        self.frustum_max = torch.tensor(self.max_dist, device='cuda')

        print(f'will be generating a total of {num_proposals} in each frustum')

        print(dict(lq=lq, uq=uq, iou_w=iou_w, dst_w=dst_w, dns_w=dns_w, \
                 min_cam_iou=min_cam_iou, size_min=size_min, size_max=size_max, ry_min=ry_min, ry_max=ry_max, cq=cq, num_mags=num_mags,
                 max_dist=max_dist, num_sizes=num_sizes, num_rotations=num_rotations, topk=topk))

        self.x_size = 25
        self.y_size = 25

        preds_path = '/home/uqdetche/GLIP/OWL_PREDEFINED_MMDETCOCO_val_kitti.coco.json'
        preds_path = model_cfg.get('PREDS_PATH', preds_path)
        print('preds_path', preds_path)

        self.image_detector = PreprocessedDetector([preds_path], class_names=class_names)

        anchors = [[3.9, 1.6, 1.56], # car
                    [6.37, 2.85, 3.19], # tram (use construction veh anchor)
                    [6.93, 2.51, 2.84], # truck
                    [6.93, 2.51, 2.84], # van (use truck)
                    [0.8, 0.6, 1.73], # person sitting (use pedestrian)

                    [1.76, 0.6, 1.73], # cyclist
                    [0.8, 0.6, 1.73], # pedestrian
        ]
        anchors = torch.tensor(anchors, dtype=torch.float32, requires_grad=False, device='cuda')
        self.anchors = anchors
        size_variations = torch.linspace(self.size_min, self.size_max, steps=self.num_sizes, device='cuda')
        base_rotations = torch.linspace(self.ry_min, self.ry_max, steps=self.num_rotations, device='cuda')
        self.base_boxes = torch.zeros((anchors.shape[0], self.num_rotations, self.num_sizes, 7), device='cuda')

        for i in range(anchors.shape[0]):
            self.base_boxes[i, :, :, [3, 4, 5]] = anchors[i]

        for i in range(self.num_rotations):
            self.base_boxes[:, i, :, -1] = base_rotations[i]

        for i, m in enumerate(size_variations):
            self.base_boxes[:, :, i, [3, 4, 5]] = self.base_boxes[:, :, i, [3, 4, 5]] * m

        self.base_corners = boxes_to_corners_3d(self.base_boxes.reshape(-1, 7)).reshape(self.anchors.shape[0], -1, 8, 3)
        self.base_boxes = self.base_boxes.reshape(self.anchors.shape[0], -1, 7)
        
        self.bev_pos = self.create_2D_grid(self.x_size, self.y_size)

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

        pc_bev_pos_range = self.point_cloud_range
        pc_bev_pos_range[0] /= 2
        pc_bev_pos_range[1] /= 2
        pc_bev_pos_range[3] /= 2
        pc_bev_pos_range[4] /= 2

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
        # print('batch_dict', batch_dict.keys())

        batch_size = batch_dict['batch_size']

        # 2d multiview detections loaded (loaded from coco jsons)
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        # print('det_boxes', det_boxes.shape)
        # print('det_scores', det_scores)
        # print('batch points', batch_dict['points'].shape)

        proposal_boxes = []
        proposal_scores = []
        frust_labels = []
        frust_batch_idx = []
        frust_scores = []

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            calib = batch_dict['calib'][b]
            calib = CalibrationTorch(calib, device=cur_points.device)

            # plane_mask = fit_plane(cur_points)
            # # print('num in plane', plane_mask.shape, plane_mask.sum(), plane_mask.sum() / cur_points.shape[0])
            # non_ground = torch.bitwise_not(plane_mask)
            # cur_points = cur_points[non_ground]
            foreground_pts = cur_points

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            batch_frusts = []
            batch_cam_boxes = []
            batch_frusts_cams = []
            batch_frusts_points = []
            batch_weighted_centres = []
            batch_bev_masks = []
            batch_scores = []
            batch_labels = []

            c = 0
            box_cam_mask = (cur_cam_idx == c)
            cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

            if cam_boxes.shape[0] > 0:
                selected = batched_nms(cam_boxes, cam_scores, cam_labels, self.nms_2d)
                cam_boxes, cam_labels, cam_scores = cam_boxes[selected], cam_labels[selected], cam_scores[selected]


            cam_points, cam_mask = self.project_to_camera(cur_points, calib)
            cam_points = cam_points[cam_mask]


            for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                if score < self.score_thr:
                    continue

                # x1, y1, w, h = box.cpu()
                # x2, y2 = x1 + w, y1 + h
                box_xyxy = box.clone()
                box_xyxy[..., 2:] += box_xyxy[..., 0:2]

                x1, y1, x2, y2 = box_xyxy

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
                    if self.search_depth is None:
                        cur_frustum_max = torch.quantile(box_points[:, 2], self.uq)
                    else:
                        # depth = self.search_depth*self.anchors[label.item() - 1, :2].norm()
                        depth = self.search_depth
                        cur_frustum_max = cur_frustum_min + depth

                    # box_centre = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2]).reshape(1, -1).cuda()
                    # dists = torch.cdist(box_centre, box_points[:, 0:2])
                    # dists_ranking = torch.softmax(- dists, dim=-1) # softmin

                    # weighted_centre_cam = dists_ranking @ box_points
                    cur_centre_z = torch.quantile(box_points[:, 2], self.cq)
                    weighted_centre_cam = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2, cur_centre_z[None].cpu()]).reshape(1, -1).cuda()

                    weighted_centre_xyz = self.get_geometry_at_image_coords(weighted_centre_cam.reshape(-1, 3), calib)

                    # weighted_centre_cam = torch.quantile(box_points, q=0.5, dim=0)
                else:
                    # print('no pts in box')
                    
                    continue
                    cur_frustum_min = self.frustum_min
                    cur_frustum_max = self.frustum_max

                    # closest points to box
                    box_centre = torch.tensor([(x1 + x2)/2, (y1 + y2)/2]).to(cam_points.device)

                    dists = torch.cdist(box_centre.reshape(1, 2), cam_points[..., :2]).reshape(-1)
                    closest_idx = torch.argmin(dists)

                    weighted_centre_xyz = cam_points[closest_idx]



                cur_frustum_max = torch.minimum(cur_frustum_max, self.frustum_max)
                cur_frustum_min = torch.maximum(cur_frustum_min, self.frustum_min)

                box_xyxy = box_xyxy.cuda()
                xyzxyz = torch.cat([box_xyxy[0][None], box_xyxy[1][None], cur_frustum_min[None], box_xyxy[2][None], box_xyxy[3][None], cur_frustum_max[None]])

                frust_box = get_cam_frustum(xyzxyz)
                frust_box = self.get_geometry_at_image_coords(frust_box, calib)

                bev_mask = self.get_frustum_bev_mask(frust_box)

                batch_frusts.append(frust_box)
                batch_cam_boxes.append(box_xyxy)
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

            N = batch_bev_masks.shape[0]
            indices = torch.arange(0, N)

            ############################################# BLENDER VISUALISATION ####################################### 
            if self.save_blender:
                frust_boxes = torch.stack(batch_frusts, dim=0).clone().cpu().numpy()

                pcl = cur_points.clone().cpu().numpy()
                print('frust_boxes', frust_boxes.shape)
                # print('pcl', pcl.shape)

                # from PIL import Image
                # img = Image.open(f'/media/bigdata/uqyluo/3D_DA/KITTI/testing/image_2/{batch_dict["frame_id"][b]}.png')
                # print(f'/media/bigdata/uqyluo/3D_DA/KITTI/testing/image_2/{batch_dict["frame_id"][b]}.png')
                # print('img', img)

                # print('images', batch_dict['images'].shape)

                # img_array = np.asarray(img)
                # img_torch = torch.tensor(img_array).permute(2, 1, 0).unsqueeze(0)
                # to_tensor = torchvision.transforms.ToTensor()
                # img_torch = to_tensor(img)
                # print('img_torch', img_torch.shape)
                # img_torch = img_torch.unsqueeze(0)

                img_torch = batch_dict['images']

                # torchvision.utils.save_image(img_torch, 'img_torch.png')

                height, width = img_torch.shape[-2:]
                print('cam_points', cam_points.shape)
                # grid = cam_points[..., :2].clone().reshape(1, 1, -1, 2)
                grid = cam_points[..., :2].clone()
                # grid[..., 0] = cam_points[..., 1]
                # grid[..., 1] = cam_points[..., 0]
                grid = grid.reshape(1, 1, -1, 2)
                grid[..., 0]  = (grid[..., 0] / width) * 2.0 - 1.0
                grid[..., 1] = (grid[..., 1] / height) * 2.0 - 1.0

                for i in [0, 1]:
                    print(f'grid{i}', grid[..., i].min(), grid[..., i].max())
                        
                sampled_image = F.grid_sample(img_torch, grid=grid)
                print('sampled', sampled_image.shape)
                sampled_image = sampled_image.reshape(3, -1).permute(1, 0)
                print('sampled', sampled_image.shape, 'pcl', pcl.shape)

                sampled_image = sampled_image.cpu().numpy()
                
                np.save('kitti_vis_col', sampled_image)
                np.save('kitti_vis_pcl', pcl)
                np.save('kitt_vis_frust_boxes', frust_boxes)
            ############################################# BLENDER VISUALISATION ####################################### 

            # frustum nms
            # indices, combined = self.frustum_bev_nms(batch_bev_masks, batch_scores, batch_labels, nms_thresh=self.nms_3d)

            # combine kept frustums with the ones that were removed (e.g. if an object is over two views, need to combine the frustums)
            # for valid_idx in indices:
            #     valid_idx = valid_idx.item()

            #     if len(combined[valid_idx]) == 0:
            #         continue

            #     base_frust = batch_frusts[valid_idx].cpu()

            #     base_centre = base_frust.mean(dim=0)
            #     base_dists = torch.cdist(base_frust, base_centre.reshape(1, 3)).reshape(-1)

            #     for comb_idx in combined[valid_idx]:
            #         comb_frust = batch_frusts[comb_idx].cpu()
            #         comb_dists = torch.cdist(comb_frust, base_centre.reshape(1, 3)).reshape(-1)

            #         # replace with corners that are further away, enlarging the frustum
            #         further_dists = comb_dists > base_dists
            #         base_frust[further_dists] = comb_frust[further_dists]

            #         # recalculate
            #         # base_centre = base_frust.mean(dim=0)
            #         base_dists = torch.cdist(base_frust, base_centre.reshape(1, 3)).reshape(-1)
            #     batch_frusts[valid_idx] = base_frust.to(batch_frusts[valid_idx].device)

            for i, (frust_box, c, box, box_points, weighted_centre_xyz, label, score) in \
                    enumerate(zip(batch_frusts, batch_frusts_cams, batch_cam_boxes, \
                                                        batch_frusts_points, batch_weighted_centres, batch_labels, batch_scores)):
                if i not in indices:
                    continue

                # project image points back to lidar
                box_points_xyz = self.get_geometry_at_image_coords(box_points, calib)

                # clamp points
                if self.clamp_bottom > 0:
                    for d in range(3):
                        f1, f2 = box_points_xyz[..., d].min(), box_points_xyz[..., d].max()
                        f1_, f2_ = frust_box[..., d].min(), frust_box[..., d].max(),

                        f1 = torch.maximum(f1, f1_)
                        f2 = torch.minimum(f2, f2_)

                        frust_box[..., d] = frust_box[..., d].clamp(f1, f2)

                frust_bev_box = torch.cat([frust_box[[2*i, 2*i+1], :].mean(dim=0)[None, :] for i in range(4)], dim=0)

                # magnitude of interpolating between closest frustum point to furtherest (producing boxes at different search_depths)
                if self.num_mags > 0:
                    mags = torch.linspace(self.mags_min, self.mags_max, self.num_mags, device='cuda').reshape(-1, 1).repeat(1, 3)
                else:
                    mags = torch.zeros((1, 3), dtype=torch.float32, device='cuda')

                frust_center_close = frust_bev_box[:2, :].mean(dim=0)
                frust_center_far = frust_bev_box[2:, :].mean(dim=0)

                center_vec = (frust_center_far - frust_center_close)

                # print('center_vec', center_vec)
                # print('mags', mags)

                if self.search_depth is not None:
                    center_vec = (center_vec / center_vec.norm()) * self.search_depth #*self.anchors[label.item() - 1, :2].norm()

                if not self.rand_center:
                    bev_pts_xyz = frust_center_close.reshape(1, 3) + center_vec * mags
                else:
                    bev_pts_xyz = weighted_centre_xyz.reshape(1, 3) + torch.randn((self.num_mags, 3), dtype=torch.float32, device=weighted_centre_xyz.device)

                # bev_pts_xyz = frust_center_close.reshape(1, 3) + center_vec * mags

                curr_corner_proposals = self.base_corners[label.item() - 1].clone()
                curr_corner_proposals = curr_corner_proposals.unsqueeze(0)
                curr_corner_proposals = curr_corner_proposals + bev_pts_xyz[:, None, None, :]
                
                # box version of the corners -> should be [N, 7]
                curr_box_proposals = self.base_boxes[label.item() - 1].clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1) # [M, N, 7] M = bev_pts_xyz.shape[0]
                curr_box_proposals[..., 0:3] = curr_box_proposals[..., 0:3] + bev_pts_xyz[:, None, :]
                
                curr_corner_proposals = curr_corner_proposals.reshape(-1, 8, 3)
                curr_box_proposals = curr_box_proposals.reshape(-1, 7)

                # now move back so the front of the box is now where the centre was
                closest_ranking = torch.softmax(- curr_corner_proposals.clone().norm(dim=2), dim=1) # softmin
                weighted_front_centres = (closest_ranking.reshape(-1, 8, 1) * curr_corner_proposals).sum(dim=1)
                
                front_to_centre = (curr_box_proposals[..., 0:3] - weighted_front_centres)
                curr_box_proposals[..., 0:3] += front_to_centre
                curr_corner_proposals += front_to_centre.reshape(-1, 1, 3)

                # filter those too far away
                dist_to_origin = weighted_front_centres.norm(dim=-1)
                valid_proposals = dist_to_origin < self.max_dist
                
                if valid_proposals.sum() == 0:
                    continue

                curr_corner_proposals = curr_corner_proposals[valid_proposals]
                curr_box_proposals = curr_box_proposals[valid_proposals]
                weighted_front_centres = weighted_front_centres[valid_proposals]

                # project all the proposed boxes to the image (so we can calculate the IOU with 2D prediction)
                image_pos, _ = self.project_to_camera(curr_corner_proposals.reshape(-1, 3), calib)
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

                if weighted_centre_xyz is not None:
                    dists = torch.cdist(weighted_front_centres.reshape(-1, 3), weighted_centre_xyz.reshape(1, 3)).reshape(-1)
                    dists_ranked = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
                    dists_ranked = 1 - dists_ranked

                    dists_ranked = dists_ranked.cpu()
                else:
                    dists_ranked = torch.ones(ious.shape[0])

                # reject under iou lower bound
                valid_proposals = ious > self.min_cam_iou
                dists_ranked = dists_ranked[valid_proposals]
                curr_corner_proposals = curr_corner_proposals[valid_proposals]
                curr_box_proposals = curr_box_proposals[valid_proposals]
                image_pos = image_pos[valid_proposals]
                proj_boxes = proj_boxes[valid_proposals]
                ious = ious[valid_proposals]

                if ious.shape[0] == 0:
                    continue
                
                # calculate how many in each proposal
                num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
                if box_points_xyz.shape[0] > 0:
                    point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(box_points_xyz.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
                    point_box_indices = point_box_indices.reshape(-1)
                    
                    for i in range(curr_box_proposals.shape[0]):
                        num_pts_in_boxes[i] = (point_box_indices == i).sum()

                soft_densities = (num_pts_in_boxes) / (num_pts_in_boxes.sum() + 1e-8)
                # second_stage_scores = ((1.0 - self.dns_w) + soft_densities * self.dns_w) * ((1.0 - self.iou_w) + ious * self.iou_w) * ((1.0 - self.dst_w) + dists_ranked.cpu() * self.dst_w)
                second_stage_scores = self.dns_w + soft_densities + self.iou_w * ious + dists_ranked.cpu() * self.dst_w

                # NMS and ordering for topk
                selected, _ = iou3d_nms_utils.nms_normal_gpu(curr_box_proposals, second_stage_scores, thresh=self.nms_normal)
                curr_box_proposals = curr_box_proposals[selected].contiguous()
                curr_corner_proposals = curr_corner_proposals[selected]
                ious = ious[selected]
                second_stage_scores = second_stage_scores[selected]

                # filter topk proposals (already in order from NMS)
                valid_proposals = slice(min(self.topk, second_stage_scores.shape[0]))
                curr_corner_proposals = curr_corner_proposals[valid_proposals]
                curr_box_proposals = curr_box_proposals[valid_proposals]
                ious = ious[valid_proposals]
                second_stage_scores = second_stage_scores[valid_proposals]

                for x in second_stage_scores: # repeat if k > 1 (topk)
                    frust_scores.append(score)
                    frust_batch_idx.append(b)
                    frust_labels.append(label)
                proposal_boxes.append(curr_box_proposals.reshape(-1, 7))
                proposal_scores.append(second_stage_scores)

        if len(proposal_boxes) > 0:
            proposal_boxes = torch.cat(proposal_boxes, dim=0).reshape(-1, 7)
            proposal_scores = torch.cat(proposal_scores, dim=0)
            frust_labels = torch.tensor(frust_labels, dtype=torch.long)
            frust_scores = torch.tensor(frust_scores)
            frust_batch_idx = torch.tensor(frust_batch_idx, dtype=torch.long)
            assert frust_scores.shape[0] == frust_labels.shape[0] and proposal_boxes.shape[0] == frust_scores.shape[0], \
                f"boxes, labels, scores shapes {proposal_boxes.shape}, {frust_labels.shape}, {frust_scores.shape}"
        else:
            proposal_boxes = torch.zeros((0, 7), device='cuda')
            frust_labels = torch.zeros((0), dtype=torch.long)
            frust_scores = torch.zeros((0))
            frust_batch_idx = torch.zeros((0), dtype=torch.long)

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

    def project_to_camera(self, points, calib: CalibrationTorch):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()
        
        cur_coords, cur_depth = calib.lidar_to_img(cur_coords[..., :3])
        cur_coords = torch.cat((cur_coords, cur_depth.unsqueeze(1)), dim=-1)

        on_img = torch.ones(cur_coords.shape[:-1], dtype=torch.bool)

        return cur_coords, on_img

    def get_geometry_at_image_coords(self, image_coords, calib: CalibrationTorch):

        pts_rect = calib.img_to_rect(image_coords[..., 0], image_coords[..., 1], image_coords[..., 2])

        return calib.rect_to_lidar(pts_rect)

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
            mask = (proposed_batch_idx == k)

            # ret_dict.append()
            # print('boxes', proposed_boxes[mask].shape)
            # print('scores', proposed_scores[mask].shape)
            # print('labels', proposed_labels[mask].shape)

            ret_dict[k]['pred_boxes'] = proposed_boxes[mask]#.cpu()
            ret_dict[k]['pred_scores'] = proposed_scores[mask]
            ret_dict[k]['pred_labels'] = proposed_labels[mask].int() # NO PLUS ONE, DONE IN PREPROCESSED_DETECTOR

        return ret_dict 
