import copy
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

from pcdet.utils.box_utils import boxes_to_corners_3d
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from torchvision.ops import box_iou, batched_nms, nms
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from ..model_utils import model_nms_utils
import time

from ..preprocessed_detector import PreprocessedDetector

BOX_FACE_IDX = [
    # pos x
    [0, 1, 4, 5],
    # neg x
    [2, 3, 6, 7],
    # neg y
    [1, 2, 5, 6],
    # pos y
    [0, 3, 4, 7]
]

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

# https://math.stackexchange.com/questions/100761/how-do-i-find-the-projection-of-a-point-onto-a-plane/100766#100766
def proj_to_plane_t(n: Tensor, pt: Tensor, pts: Tensor) -> Tensor:
    a, b, c = n[..., 0], n[..., 1], n[..., 2]
    x, y, z = pt[..., 0], pt[..., 1], pt[..., 2]
    d, e, f = pts[..., 0], pts[..., 1], pts[..., 2]

    t = (a*d - a*x + b*e - b*y + c*f - c*z)
    t = t / (a**2 + b**2 + c**2)

    return t
    # proj = [x + t*a, y + t*b, z + t*c]
    # print('proj', [x.shape for x in proj])
    # return torch.cat(proj, dim=-1)

@torch.enable_grad()
def solve_centre_og(points, curr_est, anchor, steps=10, num_mags=10, lr=0.5):
    loss_fn = nn.MSELoss()
    # loss_fn = nn.SmoothL1Loss()
    anchor_r = anchor.norm()# / 2.0

    def centre_cost(centre, pred_axis1: Tensor, pred_axis21: Tensor, pred_axis22: Tensor=None):
        inlier_mask = (points - centre.reshape(1, 3)).norm(dim=-1) < anchor_r
        inlier_points = points[inlier_mask]
        # inlier_points = points

        dists_axis1 = torch.cdist(inlier_points, pred_axis1)
        if pred_axis22 is not None:
            dists_axis21 = torch.cdist(inlier_points, pred_axis21)
            dists_axis22 = torch.cdist(inlier_points, pred_axis22)

            dists_axis21 = torch.min(dists_axis21, dim=1).values.reshape(-1, 1)
            dists_axis22 = torch.min(dists_axis22, dim=1).values.reshape(-1, 1)
            dists_axis2 = torch.minimum(dists_axis21, dists_axis22)
        else:
            dists_axis2 = torch.cdist(inlier_points, pred_axis21)
            dists_axis2 = torch.min(dists_axis2, dim=1).values.reshape(-1, 1)

        dists_axis1 = torch.min(dists_axis1, dim=1).values.reshape(-1, 1)

        dists = torch.cat((dists_axis1, dists_axis2), dim=-1)
        dists_min = torch.min(dists, dim=-1)

        # can be inside -> use relu?
        err = (dists_min.values - anchor[dists_min.indices]/2.0)

        return err.pow(2).mean()
        # return err.abs().mean()        
        # print('indices', dists_min.indices)
        # return loss_fn(dists_min.values, anchor[dists_min.indices]/2.0)

    # curr_est = points.mean(dim=0)
    centre = torch.nn.parameter.Parameter(data=curr_est, requires_grad=True)

    dir1_data = torch.ones_like(curr_est)
    dir1_data = dir1_data / dir1_data.norm(dim=-1, keepdim=True)
    dir1 = torch.nn.parameter.Parameter(data=dir1_data, requires_grad=True)

    # y = x, x = -y
    orthog_matrix = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], device=points.device, dtype=points.dtype)

    mags1 = torch.linspace(0, anchor[0]/2, steps=num_mags, device=points.device)
    mags2 = torch.linspace(0, anchor[1]/2, steps=num_mags, device=points.device)

    for i in range(steps):
        valid_centre = centre.detach().clone()

        dir1 = dir1 / dir1.norm(dim=-1, keepdim=True)

        # dir2 is an orthogonal direction to dir1 (in  x,y)
        dir2 = dir1 @ orthog_matrix
        dir3 = (-1.0) * dir2

        # print('dir1, dir2, dir3', dir1, dir2, dir3)

        pred_axis1 = centre.reshape(1, 3) + mags1.reshape(-1, 1) * dir1.reshape(1, -1) 
        pred_axis2 = centre.reshape(1, 3) + mags2.reshape(-1, 1) * dir2.reshape(1, -1) 
        pred_axis3 = centre.reshape(1, 3) + mags2.reshape(-1, 1) * dir3.reshape(1, -1) 

        cost = centre_cost(centre, pred_axis1, pred_axis2, pred_axis3)

        grad = torch.autograd.grad(cost, (centre, dir1))

        # centre = centre - centre_grad_n * cost
        # dir1 = dir1 - dir1_grad_n * cost
        centre = centre - grad[0] * lr
        dir1 = dir1 - grad[1] * lr

        d = torch.norm(centre)
        if d > 100.0:
            print('d large!', d)
            
            centre = valid_centre
            break

    centre = centre.detach().clone()
    dir1 = dir1.detach().clone()
    dir2 = dir1 @ orthog_matrix

    return centre, dir1, dir2

@torch.enable_grad()
def solve_centre_plane1(points, curr_est, anchor, steps=10, num_mags=10, lr=0.3):
    # relu_fn = nn.LeakyReLU(negative_slope=0.01)
    relu_fn = nn.ReLU()
    loss_mse = nn.L1Loss()
    
    def orthog_cost(dot_prod: Tensor):
        # dot product should be zero
        return dot_prod.pow(2)

    def cost_fn(coeffs: Tensor, centre: Tensor, dim: int):
        coeffs = coeffs.reshape(1, 3)
        centre = centre.reshape(1, 3)

        inlier_mask = (points - centre).norm(dim=-1) < (anchor.norm())
        inlier_points = points[inlier_mask]

        points_proj = (inlier_points - centre) * coeffs
        points_proj = points_proj.sum(dim=-1) # dot prod

        # can be inside -> use relu?
        err = relu_fn(points_proj.abs() - anchor[dim]/2.0)
        # print('err', err)
        centre_dist = (inlier_points - centre).norm(dim=-1)

        return err.pow(2).mean() + 0.1*centre_dist.mean()

    orthog_matrix1 = torch.tensor([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], device=points.device, dtype=points.dtype)

    orthog_matrix2 = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], device=points.device, dtype=points.dtype)

    dir1_data = torch.ones_like(curr_est)
    dir1_data = dir1_data / dir1_data.norm(dim=-1, keepdim=True)

    # dir2_data = dir1_data @ orthog_matrix1

    coeffs = torch.nn.parameter.Parameter(data=dir1_data, requires_grad=True)
    centre = torch.nn.parameter.Parameter(data=curr_est, requires_grad=True)

    sml_dim = torch.argmin(anchor[:2]).item()
    # print('smallest dim', sml_dim, anchor)
    dim2 = (sml_dim + 1) % 2

    for i in range(steps):
        coeffs = coeffs / coeffs.norm(dim=-1, keepdim=True)

        coeffs21 = coeffs @ orthog_matrix1
        coeffs22 = coeffs @ orthog_matrix2
        # print('coeffs2', coeffs2)
        # coeffs2 = coeffs2 / coeffs2.norm(dim=-1, keepdim=True)

        # coeff_dot = (coeffs * coeffs2).sum(dim=-1)

        cost = cost_fn(coeffs, centre, dim=0) + 0.5*cost_fn(coeffs21, centre, dim=1) + 0.5*cost_fn(coeffs22, centre, dim=1)
        # print('cost', cost.item(), 'lr', lr)

        grad = torch.autograd.grad(cost, (centre, coeffs))

        # centre = centre - centre_grad_n * cost
        # dir1 = dir1 - dir1_grad_n * cost
        centre = centre - grad[0] * lr
        coeffs = coeffs - grad[1] * lr
        # coeffs2 = coeffs2 - grad[2] * lr

    centre = centre.detach().clone()
    coeffs = coeffs.detach().clone()
    # coeffs2 = coeffs2.detach().clone()
    coeffs2 = coeffs @ orthog_matrix1

    return centre, coeffs, coeffs2


# latest method with 
@torch.enable_grad()
def solve_centre(points, curr_est, anchor, steps=10, num_mags=10, lr=0.1):
    relu_fn = nn.ReLU()

    def cost_fn(centre: Tensor, coeffs1: Tensor, coeffs2: Tensor):
        coeffs1 = coeffs1.reshape(1, 3)
        coeffs2 = coeffs2.reshape(1, 3)
        centre = centre.reshape(1, 3)

        inlier_mask = (points - centre).norm(dim=-1) < (anchor.norm())
        inlier_points = points[inlier_mask]
        # inlier_points = points

        centered_points = (inlier_points - centre)

        points_proj1 = centered_points * coeffs1
        points_proj1 = points_proj1.sum(dim=-1)

        points_proj2 = centered_points * coeffs2
        points_proj2 = points_proj2.sum(dim=-1)

        # each point can be assigned to either anchor[0] or anchor[1]
        a0 = anchor[0] / 2.0
        a1 = anchor[1] / 2.0
        dists_anchor0 = (points_proj1 - a0)# / a0
        dists_anchor1 = (points_proj2 - a1)# / a1

        dists = torch.stack((dists_anchor0, dists_anchor1), dim=1)

        return dists.pow(2).mean() + centered_points.norm(dim=-1).mean()

    orthog_matrix1 = torch.tensor([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], device=points.device, dtype=points.dtype)

    dir1_data = torch.ones_like(curr_est)
    dir1_data = dir1_data / dir1_data.norm(dim=-1, keepdim=True)

    coeffs = torch.nn.parameter.Parameter(data=dir1_data, requires_grad=True)
    centre = torch.nn.parameter.Parameter(data=curr_est, requires_grad=True)

    sml_dim = torch.argmin(anchor[:2]).item()
    # print('smallest dim', sml_dim, anchor)
    dim2 = (sml_dim + 1) % 2

    for i in range(steps):
        coeffs = coeffs / coeffs.norm(dim=-1, keepdim=True)

        coeffs2 = coeffs @ orthog_matrix1
        cost = cost_fn(centre, coeffs, coeffs2)

        grad = torch.autograd.grad(cost, (centre, coeffs))

        # gradient descent
        centre = centre - grad[0] * lr
        coeffs = coeffs - grad[1] * lr
    centre = centre.detach().clone()
    coeffs = coeffs.detach().clone()
    # coeffs2 = coeffs2.detach().clone()
    coeffs2 = coeffs @ orthog_matrix1

    return centre, coeffs, coeffs2
# cluster foreground / background?
def calc_angles(pts, curr_pt):
    curr_pt = curr_pt.reshape(1, -1)

    l1 = - curr_pt # to origin
    l2 = (pts - curr_pt)

    return torch.arccos((l1 @ l2.t()).reshape(-1) / (torch.norm(l1) * torch.norm(l2, dim=1) + 1e-8))

all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

class FrustumProposer(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
        iou_w=0.91, dst_w=0.57, dns_w=0.11, \
                 min_cam_iou=0.18, size_min=1.0, size_max=1.0, \
                 ry_min=0.0, ry_max=torch.pi, \
                 num_mags=10, max_dist=60, num_sizes=1, \
                 num_rotations=10, dir_iou_thr=0.3, \
                 closeq=0.33, rl=0.3, ru=0.52, num_trials=4, \
                 inlier_thr=0.69, pln_w=0.92, occl_w=0.21, topk=1,
                 angle_thresh=0.1, score_thr=0.2
    ):
        super(FrustumProposer, self).__init__()

        if model_cfg is not None:
            params_dict = model_cfg.PARAMS

            iou_w = params_dict.get('iou_w', iou_w)
            dst_w = params_dict.get('dst_w', dst_w)
            dns_w = params_dict.get('dns_w', dns_w)
            min_cam_iou = params_dict.get('min_cam_iou', min_cam_iou) 
            size_min = params_dict.get('size_min', size_min)
            size_max = params_dict.get('size_max', size_max)
            # ry_min = params_dict['ry_min']
            # ry_max = params_dict['ry_max']
            num_mags = params_dict.get('num_mags', num_mags)
            max_dist = params_dict.get('max_dist', max_dist)
            num_sizes = params_dict.get('num_sizes', num_sizes)
            num_rotations = params_dict.get('num_rotations', num_rotations)
            topk = params_dict.get('topk', topk)
            dir_iou_thr = params_dict.get('dir_iou_thr', dir_iou_thr)
            rl = params_dict.get('rl', rl)
            ru = params_dict.get('ru', ru)
            num_trials = params_dict.get('num_trials', num_trials)
            inlier_thr = params_dict.get('inlier_thr', inlier_thr)
            pln_w = params_dict.get('pln_w', pln_w)
            occl_w = params_dict.get('occl_w', occl_w)
            angle_thresh = params_dict.get('angle_thresh', angle_thresh)

        self.class_names = class_names
        # print('frustum class_names', self.class_names)
        self.class_labels = [i for i, x in enumerate(all_class_names) if x in class_names]

        self.image_order = [2, 0, 1, 5, 3, 4]
        # self.image_size = [512, 800]
        self.image_size = [900, 1600]

        self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

        self.topk = topk

        self.dir_iou_thr = dir_iou_thr
        self.rl = rl
        self.ru = ru
        self.num_trials = num_trials
        self.inlier_thr = inlier_thr

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


        self.iou_w = iou_w
        self.m_iou_w = 1.0 - iou_w
        self.dst_w = dst_w
        self.m_dst_w = 1.0 - dst_w
        self.pln_w = pln_w
        self.m_pln_w = 1.0 - pln_w
        self.occl_w = occl_w
        self.m_occl_w = 1.0 - occl_w
        self.dns_w = dns_w
        self.m_dns_w = 1.0 - dns_w

        self.angle_thresh = angle_thresh

        self.min_cam_iou = torch.tensor(min_cam_iou)
        self.max_clusters = 20

        offset_vals = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, -1),
            (0, -1, 0),
            (-1, 0, 0),
            (0, 0, 0)
        ]
        self.offset_vals_tensor = torch.tensor(offset_vals, device='cuda')

        num_proposals = self.num_rotations * len(offset_vals)

        self.frustum_min = torch.tensor(2.0, device='cuda')
        self.frustum_max = torch.tensor(self.max_dist, device='cuda')

        print(f'will be generating a total of {num_proposals} in each frustum')

        print(dict(iou_w=iou_w, dst_w=dst_w, dns_w=dns_w, \
                 min_cam_iou=min_cam_iou, size_min=size_min, size_max=size_max, ry_min=ry_min, ry_max=ry_max, num_mags=num_mags,
                 max_dist=max_dist, num_sizes=num_sizes, num_rotations=num_rotations, topk=topk))

        self.x_size = 25
        self.y_size = 25

        # preds_path = '/home/uqdetche/OpenPCDet/tools/coco_val_'
        preds_path = '/home/uqdetche/GLIP/jsons/OWL_'
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

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

    # @profile
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
        # lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        frust_colors = []
        frust_batch_idx = []
        frust_proposals_corners = []
        proposal_boxes = []
        frust_scores = []
        frust_labels = []
        frust_second_stage_scores = []
        box_points_xyzs = []
        box_points_labels_ = []

        weighted_centres = []
        medians = []

        largest_bev_union_num = 0
        largest_bev_union = None
        
        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            cur_points_dirs = cur_points.clone()#.unsqueeze(0)
            cur_points_mags = torch.norm(cur_points_dirs, dim=-1, keepdim=True)
            cur_points_dirs = cur_points_dirs / cur_points_mags


            # print('plane_coeff', plane_coeff)

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            batch_frusts = []
            batch_cam_boxes = []
            batch_frusts_cams = []
            batch_frusts_points = []
            batch_weighted_centres = []
            batch_scores = []
            batch_labels = []

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                # 2d nms
                # selected = batched_nms(cam_boxes, cam_scores, cam_labels, 0.4)
                if cam_boxes.shape[0] > 0:
                    selected = batched_nms(cam_boxes, cam_scores, cam_labels, 0.4)
                    cam_boxes, cam_labels, cam_scores = cam_boxes[selected], cam_labels[selected], cam_scores[selected]

                cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                cam_points = cam_points[cam_mask]

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < self.score_thr:
                        continue

                    if label not in self.class_labels:
                        # not in our config class_names!
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

                    if box_points.numel() == 0:
                        # print('no points!')
                        continue

                    cur_frustum_max = torch.minimum(box_points[:, 2].max(), self.frustum_max)
                    cur_frustum_min = torch.maximum(box_points[:, 2].min(), self.frustum_min)

                    box = box.cuda()
                    xyzxyz = torch.cat([box[0][None], box[1][None], cur_frustum_min[None], box[2][None], box[3][None], cur_frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    # bev_mask = self.get_frustum_bev_mask(frust_box)
                    box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    batch_frusts.append(frust_box)
                    batch_cam_boxes.append(box)
                    batch_frusts_cams.append(c)
                    # batch_frusts_points.append(box_points)
                    batch_frusts_points.append(box_points_xyz)

                    # batch_weighted_centres.append(weighted_centre_xyz)
                    # batch_bev_masks.append(bev_mask[None])
                    batch_scores.append(score[None])
                    batch_labels.append(label[None])

            if len(batch_scores) == 0:
                continue

            batch_scores = torch.cat(batch_scores, dim=0)

            keep = torch.ones_like(batch_scores, dtype=torch.bool)
            order = torch.argsort(- batch_scores)

            # camera nms
            # for i, (frust_boxi, ci, labeli) in enumerate(zip(batch_frusts, batch_frusts_cams, batch_labels)):
            for i in range(order.shape[0]):
                curr_idx = order[i].item()
                frust_boxi, ci, labeli = batch_frusts[curr_idx], batch_frusts_cams[curr_idx], batch_labels[curr_idx]

                frust_boxi_dirs = frust_boxi / torch.norm(frust_boxi, keepdim=True, dim=-1)
                min_dirsi = frust_boxi_dirs.min(dim=0).values
                max_dirsi = frust_boxi_dirs.max(dim=0).values

                # for j, (frust_boxj, cj, labelj) in enumerate(zip(batch_frusts, batch_frusts_cams, batch_labels), start=i+1):
                for j in range(i + 1, order.shape[0]):
                    other_idx = order[j].item()
                    frust_boxj, cj, labelj = batch_frusts[other_idx], batch_frusts_cams[other_idx], batch_labels[other_idx]

                    # cameras different and boxes different and labels same!
                    if labeli != labelj or ci == cj:
                    # if labeli != labelj or ci == cj or (ci, cj) not in valid_cam_nms:
                        continue

                    frust_boxj_dirs = frust_boxj / torch.norm(frust_boxj, keepdim=True, dim=-1)
                    min_dirsj = frust_boxj_dirs.min(dim=0).values
                    max_dirsj = frust_boxj_dirs.max(dim=0).values

                    int_min_dir = torch.maximum(min_dirsi, min_dirsj)
                    int_max_dir = torch.minimum(max_dirsi, max_dirsj)

                    valid = (int_max_dir > int_min_dir).all()

                    if not valid:
                        continue

                    if keep[j]:
                        # remove the j box
                        keep[j] = False

                        batch_frusts_points[curr_idx] = torch.cat([batch_frusts_points[curr_idx], batch_frusts_points[other_idx]], dim=0)

                        comb_dists = frust_boxj_dirs @ frust_boxi_dirs.permute(1, 0)
                        comb_dists = comb_dists.max(dim=1).values
                        
                        further_angles = torch.topk(comb_dists, k=4, dim=0, largest=False).indices
                        frust_boxi[further_angles] = frust_boxj[further_angles]
                        
                batch_frusts[curr_idx] = frust_boxi

            # print('NMS removed', (~keep).sum())

            # indices for nms
            indices = order[keep]

            indices_list = [i.item() for i in indices]
            batch_frusts = [batch_frusts[i] for i in indices_list]
            batch_frusts_cams = [batch_frusts_cams[i] for i in indices_list]
            batch_cam_boxes = [batch_cam_boxes[i] for i in indices_list]
            batch_frusts_points = [batch_frusts_points[i] for i in indices_list]
            batch_labels = [batch_labels[i] for i in indices_list]

            for i, (frust_box, c, box, box_points_xyz, label, cam_score) in \
                    enumerate(zip(batch_frusts, batch_frusts_cams, batch_cam_boxes, \
                                            batch_frusts_points, batch_labels, batch_scores)):

                box_points_mags = torch.norm(box_points_xyz, dim=-1, keepdim=True)
                # box_points_xyz_dirs = box_points_xyz / box_points_mags

                curr_anchor = self.anchors[int(label)]
                curr_anchor_max_diff = torch.tensor([curr_anchor[:2].max(), curr_anchor[:2].max(), curr_anchor[2]]).to(curr_anchor.device)
                curr_anchor_r = curr_anchor.pow(2).sum().sqrt()# / 2.0
                n_pts = box_points_xyz.shape[0]

                connected = torch.zeros((n_pts, n_pts), dtype=torch.bool)
                for i in range(n_pts):
                    angles = calc_angles(box_points_xyz, box_points_xyz[i])
                    dists = torch.cdist(box_points_xyz, box_points_xyz[[i]]).reshape(-1)
                    # connected[i, :] = angles > angles.quantile(0.1)
                    connected[i, :] = (angles > self.angle_thresh) & (dists < curr_anchor_r)
                    # connected[i, :] = (dists < curr_anchor_r)
                
                connected_sum = connected.sum(dim=1)

                pt_ranking = torch.softmax(connected_sum.float(), dim=0) * torch.softmax(-box_points_mags.cpu().reshape(-1), dim=0)

                # closest = torch.argmax(connected_sum)
                closest = torch.argmax(pt_ranking)
                connected_to_first = connected[closest]
                connected_box_points = box_points_xyz[connected_to_first]

                n_pts = connected_box_points.shape[0]

                if n_pts == 0:
                    connected_box_points = box_points_xyz
                    n_pts = connected_box_points.shape[0]


                dist_labels = torch.zeros((n_pts, ), dtype=torch.long)
                dists = torch.cdist(connected_box_points, connected_box_points)
                cur_indices = torch.arange(n_pts)

                curr_label = 1
                dist_labels[0] = curr_label
                # dist_labels[connected_to_first] = curr_label
                # curr_idx = cur_indices[connected_to_first][0]
                curr_idx = 0

                while (dist_labels == 0).sum() > 0:
                    not_visited_mask = (dist_labels == 0)
                    # cur_label_mask = (dist_labels == curr_label)

                    # find closest not visited 
                    closest_idx = torch.argmin(dists[curr_idx, not_visited_mask])
                    closest_idx = cur_indices[not_visited_mask][closest_idx]

                    # distance to that one
                    closest_diff = (connected_box_points[curr_idx] - connected_box_points[closest_idx]).abs()

                    # if closest_dist > anchor_r: 
                    # if (closest_diff > curr_anchor_max_diff/2.0).any():# or not connected[curr_idx, closest_idx]:
                    if (closest_diff > curr_anchor_max_diff).any():# or not connected[curr_idx, closest_idx]:
                        curr_label += 1

                    if curr_label > self.max_clusters:
                        break

                    dist_labels[closest_idx] = curr_label

                    curr_idx = closest_idx

                # print('clusters', dist_labels.max())
                if dist_labels.max() > self.max_clusters:
                    print('so many clusters :O', dist_labels.max())

                num_clusters = min(self.max_clusters, dist_labels.max())

                curr_box_proposals_list = []

                for dist_label in range(1, num_clusters + 1):
                    dist_label_mask = (dist_labels == dist_label)
                    curr_connected_box_points = connected_box_points[dist_label_mask]
                    curr_connected_box_points_dirs = curr_connected_box_points / torch.norm(curr_connected_box_points, dim=-1, keepdim=True)

                    # find centre interest area
                    # curr_centre_offs = curr_anchor_max_diff.reshape(-1, 3) * torch.sign(curr_connected_box_points)
                    curr_centre_offs = curr_connected_box_points_dirs * curr_anchor_r
                    curr_centre_offs[..., 0] = torch.clamp(curr_centre_offs[..., 0], min=-curr_anchor_max_diff[0]/2, max=curr_anchor_max_diff[0]/2)
                    curr_centre_offs[..., 1] = torch.clamp(curr_centre_offs[..., 1], min=-curr_anchor_max_diff[1]/2, max=curr_anchor_max_diff[1]/2)
                    curr_centre_offs[..., 2] = torch.clamp(curr_centre_offs[..., 2], min=-curr_anchor_max_diff[2]/2, max=curr_anchor_max_diff[2]/2)

                    curr_centre_ests = curr_connected_box_points + curr_centre_offs

                    # least_squares way
                    curr_estimate = curr_centre_ests.mean(dim=0)

                    centre, dir1, dir2 = solve_centre(curr_connected_box_points, curr_estimate, curr_anchor, steps=10, num_mags=5)

                    
                    r1 = torch.atan2(dir1[1], dir1[0])
                    curr_rotations = torch.linspace(r1 - torch.pi/2, r1 + torch.pi/2, self.num_rotations, device='cuda')

                    curr_box_proposals = torch.zeros((len(curr_rotations), len(self.offset_vals_tensor), 7), device='cuda')

                    curr_box_proposals[:, :, [3, 4, 5]] = curr_anchor
                    for j, offset in enumerate(self.offset_vals_tensor):
                        for k, ry in enumerate(curr_rotations):
                            xyz = 0.25 * offset * curr_anchor

                            xyz = common_utils.rotate_points_along_z(xyz.reshape(1, 1, 3), ry.reshape(1)).reshape(3)
                            xyz = xyz + centre

                            curr_box_proposals[k, j, 0:3] = xyz
                            curr_box_proposals[k, j, -1] = ry

                    curr_box_proposals_list.append(curr_box_proposals.reshape(-1, 7))

                    # print('curr_box_proposals', curr_box_proposals)

                curr_box_proposals = torch.cat(curr_box_proposals_list, dim=0)
                # print('box proposals', curr_box_proposals.shape)

                proposal_box, out_score = self.get_frust_proposals(curr_box_proposals, c, b, box, batch_dict, cur_points, cur_points_dirs, cur_points_mags)

                if out_score is None:
                    continue

                # for x in second_stage_scores: # repeat if second stage > 1 proposals
                for x in out_score: 
                    # frust_scores.append(x)
                    frust_scores.append(cam_score)
                    frust_batch_idx.append(b)
                    frust_labels.append(label)

                proposal_boxes.append(proposal_box.reshape(-1, 7))

        if len(proposal_boxes) > 0:
            proposal_boxes = torch.cat(proposal_boxes, dim=0).reshape(-1, 7)
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

            # frust_proposals_corners = torch.cat(frust_proposals_corners, dim=0)

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()

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


    def get_frust_proposals(self, curr_box_proposals, c, b, box, batch_dict, cur_points, cur_points_dirs, cur_points_mags):
        curr_box_proposals = curr_box_proposals.reshape(-1, 7)
        curr_corner_proposals = boxes_to_corners_3d(curr_box_proposals).reshape(-1, 8, 3)

        # now move back so the front of the box is now where the centre was
        closest_ranking = torch.softmax(- curr_corner_proposals.clone().norm(dim=2), dim=1) # softmin
        weighted_front_centres = (closest_ranking.reshape(-1, 8, 1) * curr_corner_proposals).sum(dim=1)

        image_pos, _ = self.project_to_camera(batch_dict, curr_corner_proposals.reshape(-1, 3), b, c)
        image_pos = image_pos[..., :2].reshape(-1, 8, 2)

        # filter those too far away
        dist_to_origin = weighted_front_centres.norm(dim=-1)
        valid_proposals = dist_to_origin < self.max_dist
        
        if valid_proposals.sum() == 0:
            # print('none are valid!', dist_to_origin)
            return None, None

        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        image_pos = image_pos[valid_proposals]
        weighted_front_centres = weighted_front_centres[valid_proposals]

        image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
        image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

        x1, y1, x2, y2 = image_pos[..., 0].min(dim=-1).values, image_pos[..., 1].min(dim=-1).values, image_pos[..., 0].max(dim=-1).values, image_pos[..., 1].max(dim=-1).values

        proj_boxes = torch.zeros((x1.shape[0], 4))
        proj_boxes[:, 0] = x1
        proj_boxes[:, 1] = y1
        proj_boxes[:, 2] = x2
        proj_boxes[:, 3] = y2

        ious = box_iou(proj_boxes, box.cpu().reshape(1, -1)).reshape(-1)
        # print('max iou', ious.max())

        # dists = (weighted_front_centres.reshape(-1, 1, 3) - bev_pts_xyz.reshape(1, -1, 3)).pow(2).sum(dim=-1).sqrt().min(dim=1).values
        # dists = torch.cdist(weighted_front_centres.reshape(-1, 3), weighted_centre_xyz.reshape(1, 3)).reshape(-1)
        # dists_ranked = torch.softmax(-dists, dim=0).cpu()


        # reject under iou lower bound
        # valid_proposals = ious >= torch.maximum(self.min_cam_iou, ious.mean())
        valid_proposals = ious >= self.min_cam_iou
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        ious = ious[valid_proposals]
        # dists_ranked = dists_ranked[valid_proposals]

        if ious.shape[0] == 0:
            # print('non valid after iou')
            return None, None

        # find point cloud points of which their ray collides with the box (at some point)
        curr_proposal_dirs = curr_corner_proposals.clone()
        curr_proposal_dists = torch.norm(curr_proposal_dirs, dim=-1, keepdim=True)
        curr_proposal_dirs = curr_proposal_dirs / curr_proposal_dists

        min_dirs = curr_proposal_dirs.min(dim=1).values
        max_dirs = curr_proposal_dirs.max(dim=1).values

        occlusion_scores = torch.zeros(curr_box_proposals.shape[0])
        plane_scores = torch.zeros_like(occlusion_scores)

        for i, (min_dir, max_dir) in enumerate(zip(min_dirs, max_dirs)):
            dmin, dmax = curr_proposal_dists[i].min(), curr_proposal_dists[i].max()

            pt_masks = (cur_points_dirs >= min_dir) & (cur_points_dirs <= max_dir) & (cur_points_mags >= dmin) #& (cur_points_mags <= dmax)
            pt_masks = pt_masks.all(dim=-1)

            real_points = cur_points[pt_masks]

            if real_points.shape[0] == 0:
                # occlusion_scores[i] = 1.0 # no real points
                continue

            real_dists = torch.norm(real_points, dim=-1).unsqueeze(0)
            real_dirs = cur_points_dirs[pt_masks]

            rs = torch.linspace(self.rl, self.ru, self.num_trials, device=curr_proposal_dirs.device).reshape(-1, 1)
            # rs = torch.linspace(0.0, 1.0, num_trials, device=curr_proposal_dirs.device).reshape(-1, 1)
            # mags = real_dists + (real_dists - dmin) * rs
            mags = real_dists * rs
            mags = mags.unsqueeze(-1)
            empty_points = real_dirs.unsqueeze(0) * mags
            empty_points = empty_points.reshape(-1, 3)

            curr_box_proposal = curr_box_proposals[i].reshape(1, -1, 7)
            curr_corner_proposal = curr_corner_proposals[i].reshape(8, 3)
            box_centre = curr_box_proposal[0, 0, :3]
            box_centre_vec = box_centre / torch.norm(box_centre, dim=-1, keepdim=True)

            max_face_score = 0.0

            for fi, face_idx in enumerate(BOX_FACE_IDX):
                face_centre = curr_corner_proposal[face_idx].mean(dim=0)
                face_proposal_dirs = curr_proposal_dirs[i, face_idx]
                # face_min_dir = face_proposal_dirs.min(dim=0).values
                # face_max_dir = face_proposal_dirs.max(dim=0).values

                face_vec = face_centre - box_centre
                face_vec_mag = torch.norm(face_vec, dim=-1, keepdim=True)
                face_norm = face_vec / face_vec_mag

                facing_cam = (box_centre_vec * face_norm).sum(dim=-1) < 0.0

                if facing_cam:
                    # project to plane :)

                    # face_pt_masks = (cur_points_dirs >= face_min_dir) & (cur_points_dirs <= face_max_dir)
                    # face_pt_masks = face_pt_masks.all(dim=-1)
                    # face_points = cur_points[face_pt_masks]

                    t = proj_to_plane_t(face_norm, face_centre, real_points)

                    if t.numel() > 0:
                        inliers = t.abs() < self.inlier_thr
                        curr_face_score = inliers.sum().cpu() / real_points.shape[0]
                        plane_scores[i] += curr_face_score

                        # if curr_face_score > max_face_score:
                            # max_face_score = curr_face_score

                        # plane_scores[i] += t.abs().mean().cpu()

            plane_scores[i] /= 2.0
            # plane_scores[i] = max_face_score

            real_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(real_points.unsqueeze(0), curr_box_proposal)
            real_mask = (real_point_box_indices == 0).reshape(-1)

            num_in_points = real_mask.sum()
            # now check if the ray to each out point passes through the box
            out_points = real_points[~real_mask]

            if out_points.shape[0] == 0:
                occlusion_scores[i] = - 2*num_in_points #in_points.shape[0] / real_points.shape[0] # all real points
                continue
            
            out_mags = torch.norm(out_points, dim=-1, keepdim=True)
            out_dirs = out_points / out_mags

            # remove out points before box
            out_mask = (out_mags >= dmin).reshape(-1)
            out_points = out_points[out_mask]
            out_dirs = out_dirs[out_mask]
            out_mags = out_mags[out_mask]

            trial_mags = torch.linspace(dmin, dmax, self.num_trials, device=curr_proposal_dirs.device).reshape(-1, 1, 1)
            trial_pts = out_dirs.unsqueeze(0) * trial_mags
            trial_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(trial_pts.reshape(1, -1, 3), curr_box_proposal)
            trial_box_indices = trial_box_indices.reshape(self.num_trials, -1)
            trial_in_mask = (trial_box_indices == 0).any(dim=0)
            trial_in_mask = trial_in_mask.reshape(-1)
            num_invalid_occl = trial_in_mask.sum()

            empty_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(empty_points.unsqueeze(0), curr_box_proposal)
            empty_mask = (empty_point_box_indices == 0).reshape(-1)
            num_empty = empty_mask.sum()

            # empty_ratio = num_empty / empty_mask.shape[0]
            # occl_ratio = num_invalid_occl / trial_in_mask.shape[0]

            score = (num_invalid_occl + num_empty/self.num_trials) - 2*num_in_points

            occlusion_scores[i] = score

        num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
        point_box_indices = point_box_indices.reshape(-1)
        
        for i in range(curr_box_proposals.shape[0]):
            num_pts_in_boxes[i] = (point_box_indices == i).sum()

        occlusion_scores_ranked = torch.softmax(-occlusion_scores, dim=0)
        plane_scores_ranked = torch.softmax(plane_scores, dim=0)
        # soft_densities = torch.softmax(num_pts_in_boxes, dim=0)
        soft_densities = (num_pts_in_boxes - num_pts_in_boxes.min()) / (num_pts_in_boxes.max() - num_pts_in_boxes.min())
        # ious_ranked = torch.softmax(ious, dim=0)

        # second_stage_scores = ((1.0 - self.dns_w) + soft_densities * self.dns_w) * ((1.0 - self.iou_w) + ious * self.iou_w) * ((1.0 - self.dst_w) + dists_ranked.cpu() * self.dst_w) * occlusion_scores_ranked
        second_stage_scores = (self.m_occl_w + self.occl_w * occlusion_scores_ranked) * (self.m_iou_w + self.iou_w * ious) \
                * (self.m_pln_w + self.pln_w * plane_scores_ranked)  * (self.m_dns_w + self.dns_w * soft_densities)
                # * (self.m_dst_w + self.dst_w * dists_ranked)

        # second_stage_scores = (self.m_occl_w + self.occl_w * occlusion_scores_ranked) \
                # * (self.m_iou_w + self.iou_w * ious) \
                # * (self.m_pln_w + self.pln_w * plane_scores_ranked) \
                # * (self.m_dns_w + self.dns_w * soft_densities) \
                # + (self.m_dst_w + self.dst_w * dists_ranked)
        # second_stage_scores = (0.1 + 0.9 * occlusion_scores_ranked) * (0.1 + 0.9 * ious) * (0.5 + 0.9 * plane_scores_ranked) * (0.1 + 0.9 * soft_densities)
        second_stage_scores = second_stage_scores / second_stage_scores.sum()
        # second_stage_scores = plane_scores_ranked * occlusion_scores_ranked * ious * soft_densities * dists_ranked.cpu()
        # print('second stage', second_stage_scores)

        if self.topk > 1:
            valid_proposals = torch.topk(second_stage_scores, k=min(self.topk, second_stage_scores.shape[0])).indices
        else:
            valid_proposals = torch.argmax(second_stage_scores).reshape(-1)
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        second_stage_scores = second_stage_scores[valid_proposals]
        ious = ious[valid_proposals]

        return curr_box_proposals, second_stage_scores

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

            batch_boxes = proposed_boxes[mask]
            batch_scores = proposed_scores[mask]
            batch_labels = proposed_labels[mask]

            # if batch_boxes.shape[0] > 1:
            #     selected, _ = iou3d_nms_utils.nms_normal_gpu(batch_boxes, batch_scores, thresh=0.05)
            #     # selected, _ = iou3d_nms_utils.nms_gpu(batch_boxes, batch_scores, thresh=0.01)
            #     print('selected', selected.shape, batch_boxes.shape)
            #     batch_boxes = batch_boxes[selected]
            #     batch_scores = batch_scores[selected]
            #     batch_labels = batch_labels[selected]

            batch_boxes = batch_boxes.to(dtype=torch.float)
            batch_scores = batch_scores.to(dtype=torch.float)

            ret_dict[k]['pred_boxes'] = batch_boxes#.cpu()
            ret_dict[k]['pred_scores'] = batch_scores
            ret_dict[k]['pred_labels'] = batch_labels.int() + 1

        return ret_dict 
