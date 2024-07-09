import argparse
import glob
from pathlib import Path

import random

import numpy as np
import torch
from torch import nn, Tensor
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from typing import Dict

from matplotlib import pyplot as plt

from pcdet.models.preprocessed_detector import PreprocessedDetector
from pcdet.utils.box_utils import boxes_to_corners_3d
from torchvision.ops import box_iou, batched_nms, nms
from matplotlib import pyplot as plt
from pcdet.models.dense_heads.transfusion_head_2D_proposals import PALETTE, draw_corners_on_image
# from demo_pyplot import view_points
import json

from scipy.optimize import least_squares
from sklearn.cluster import HDBSCAN

from enum import Enum

# class syntax

def batched_distance_nms(boxes, scores, labels):
    dists = torch.cdist(boxes[..., :3], boxes[..., :3])

    order = torch.argsort(-scores, dim=0)
    keep = torch.ones_like(order, dtype=torch.bool)

    for i in range(len(boxes)):
        curr_idx = order[i]
        curr_lbl = labels[curr_idx]
            
        if not keep[i]:
            continue

        for j in range(i + 1, len(boxes)):
            oth_idx = order[j]
            oth_lbl = labels[oth_idx]
            anch_r = boxes[oth_idx, 3:6].norm() / 2.0

            if curr_lbl == oth_lbl and keep[j] and dists[curr_idx, oth_idx] <= anch_r:
                keep[j] = False

    print('removed', (keep.numel() - keep.sum()), 'orig', keep.numel())

    return order[keep]


class ClusterMethod(Enum):
    HDB = 0
    ANGLE = 1
    
class HDBSCANCluster:
    def __init__(self, min_cluster_size: int = 5):
        self.hdb = HDBSCAN(min_cluster_size=min_cluster_size)

    def __call__(self, frust_points: Tensor, curr_anchor: Tensor):
        points_np = frust_points.cpu().numpy()

        # clustering
        if points_np.shape[0] >= self.hdb.min_cluster_size:
            labels = self.hdb.fit_predict(points_np)
        else:
            labels = np.ones((points_np.shape[0]), dtype=int)
        return torch.from_numpy(labels)

class AngleClusterer:
    angle_thr: float = torch.pi/2.0
    # max_clusters: int = 5
    invalid_label: int = -1

    def __init__(self):
        pass

    def __call__(self, frust_points: Tensor, curr_anchor: Tensor, max_clusters: int = 2):
        curr_anchor_r = curr_anchor.norm()
        curr_anchor_max_diff = torch.tensor([curr_anchor[:2].max(), curr_anchor[:2].max(), curr_anchor[2]]).to(curr_anchor.device)

        n_pts = frust_points.shape[0]

        angles = frust_points.new_zeros((n_pts, n_pts))
        for i in range(n_pts):
            angles[i] = calc_angles(frust_points, frust_points[i]).reshape(-1)

        dist_labels = torch.zeros((n_pts, ), dtype=torch.long)
        dist_labels[:] = self.invalid_label

        dists = torch.cdist(frust_points, frust_points)
        cur_indices = torch.arange(n_pts)

        curr_label = 0
        curr_idx = 0
        dist_labels[curr_idx] = curr_label

        while (dist_labels == self.invalid_label).sum() > 0:
            not_visited_mask = (dist_labels == self.invalid_label)

            # find closest not visited 
            closest_idx = torch.argmin(dists[curr_idx, not_visited_mask])
            closest_idx = cur_indices[not_visited_mask][closest_idx]

            # distance to that one
            closest_diff = (frust_points[curr_idx] - frust_points[closest_idx]).abs()

            # largest angle from current point to points in current cluster (larger -> closer)
            curr_angles = angles[curr_idx, not_visited_mask]
            closest_angle = curr_angles.max()

            if (closest_diff > curr_anchor_max_diff).any() or closest_angle < self.angle_thr:
                curr_label += 1

            if curr_label > max_clusters:
                break

            dist_labels[closest_idx] = curr_label

            curr_idx = closest_idx

        return dist_labels

def norm01(vals):
    return (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

class CombinedAngleClusterer:
    angle_thr: float = torch.pi/2.0
    # angle_thr: float = 0.4*torch.pi
    # angle_thr: 0.03
    # max_clusters: int = 5
    invalid_label: int = -1

    def __init__(self, angle_thr: float = torch.pi/2.0):
        self.angle_thr = angle_thr

    def __call__(self, frust_points: Tensor, frust_labels: Tensor, anchors: Tensor):
        # curr_anchor_r = curr_anchor.norm()
        # curr_anchor_max_diff = torch.tensor([curr_anchor[:2].max(), curr_anchor[:2].max(), curr_anchor[2]]).to(frust_points.device
        curr_anchor_max_diffs = anchors.clone()

        xy_max = anchors[..., :2].max(dim=1).values
        curr_anchor_max_diffs[..., 0] = xy_max
        curr_anchor_max_diffs[..., 1] = xy_max

        n_pts = frust_points.shape[0]

        angles = frust_points.new_zeros((n_pts, n_pts))
        for i in range(n_pts):
            angles[i] = calc_angles(frust_points, frust_points[i]).reshape(-1)

        dist_labels = torch.zeros((n_pts, ), dtype=torch.long)
        dist_labels[:] = self.invalid_label

        dists = torch.cdist(frust_points, frust_points)
        cur_indices = torch.arange(n_pts)

        curr_clust_label = 0
        curr_idx = 0
        dist_labels[curr_idx] = curr_clust_label

        cluster_classes = dict()

        while (dist_labels == self.invalid_label).sum() > 0:
            not_visited_mask = (dist_labels == self.invalid_label)
            curr_cluster_mask = (dist_labels == curr_clust_label)
            unvisited_labels = frust_labels[not_visited_mask]

            # set cluster label
            if curr_clust_label not in cluster_classes.keys():
                cluster_classes[curr_clust_label] = frust_labels[curr_idx].item()

            curr_centre = frust_points[curr_cluster_mask].mean(dim=0)

            dist_cost = norm01(dists[curr_idx, not_visited_mask])
            label_cost = (unvisited_labels - frust_labels[curr_idx]).abs().clamp(max=1.0)

            # print('dist, angle, label', dist_cost.shape, angle_cost.shape, label_cost.shape)
            # print('dist, angle, label', dist_cost.device, angle_cost.device, label_cost.device)

            closest_cost = dist_cost + label_cost

            # find closest not visited 
            closest_idx = torch.argmin(closest_cost)
            closest_idx = cur_indices[not_visited_mask][closest_idx]

            # absolute distance to the closest point
            closest_diff = (frust_points[curr_idx] - frust_points[closest_idx])
            closest_abs_diff = closest_diff.abs()
            diff_norm = closest_diff.norm()

            centre_distance = (curr_centre - frust_points[closest_idx]).norm()

            # largest angle from current point to points in current cluster (larger -> closer)
            curr_angles = angles[curr_idx, curr_cluster_mask]
            # if curr_angles.numel() == 1:
                # print('curr_angles', curr_angles)
            # exit()
            closest_angle = curr_angles.max()

            # select curr anchor
            curr_anchor = anchors[frust_labels[curr_idx]]
            curr_anchor_max_diff = curr_anchor_max_diffs[frust_labels[curr_idx]]
            curr_anchor_r = curr_anchor.norm()
            curr_anchor_r2 = curr_anchor_r / 2.0

            if (closest_abs_diff > curr_anchor_max_diff).any() or closest_angle < self.angle_thr \
                    or (frust_labels[closest_idx] - frust_labels[curr_idx]).abs() > 0 or diff_norm > curr_anchor_r2 or centre_distance > curr_anchor_r2:
                curr_clust_label += 1

            # if curr_clust_label > max_clusters:
                # break

            dist_labels[closest_idx] = curr_clust_label

            curr_idx = closest_idx

        # num clusters
        num_clusters = dist_labels.max()

        # indices = torch.arange(0, num_clusters, 1)
        indices = torch.arange(num_clusters)
        keep = frust_points.new_ones(num_clusters, dtype=torch.bool)
        num_per_cluster = frust_points.new_zeros(num_clusters, dtype=torch.long)
        cluster_centres = frust_points.new_zeros((num_clusters, 3))

        for i in range(num_clusters):
            cluster_mask = (dist_labels == i)
            num_per_cluster[i] = cluster_mask.sum()
            cluster_centres[i] = frust_points[cluster_mask].mean(dim=0)

        order = torch.argsort(- num_per_cluster)

        for i in range(order.shape[0]):
            idx = order[i].item()

            if not keep[i]:
                continue

            # find closest to idx
            curr_centre = cluster_centres[idx]
            curr_num = num_per_cluster[idx]
            curr_class = cluster_classes[idx]
            curr_anchor = anchors[curr_class]

            curr_anchor_r2 = curr_anchor.norm() / 2.0

            for j in range(i + 1, order.shape[0]):
                oidx = order[j].item()

                other_class = cluster_classes[oidx]

                if curr_class != other_class:
                    continue

                other_centre = cluster_centres[oidx]

                # guaranteed to be <= current
                other_num = num_per_cluster[oidx]

                # print('current, other num', curr_num, other_num)

                centre_distance = (curr_centre - other_centre).norm()

                if keep[j] and centre_distance < curr_anchor_r2:
                    keep[j] = False
                    num_per_cluster[idx] += other_num
                    dist_labels[(dist_labels == oidx)] = idx # relabel


        return dist_labels

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208)]

# with open('/home/uqdetche/OpenPCDet/tools/coco_classes_91.txt', 'r') as f:
#     lines = f.readlines()

#     coco_classes = [['background']]
#     coco_classes.extend(lines)

PALETTE = [[x/255 for x in y] for y in PALETTE]

all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
all_colors = ['red', 'red', 'red', 'red', ]
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def connected_corners():
    corners = [(0, 5), (1, 4)]

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        corners.append((i, j))
        i, j = k + 4, (k + 1) % 4 + 4
        corners.append((i, j))

        i, j = k, k + 4
        corners.append((i, j))

    return corners

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

all_face_idx = [idx for FACE_IDX in BOX_FACE_IDX for idx in FACE_IDX]

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

def boxes_to_corners_3d_offset(boxes3d, offsets):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    # boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    nbox = boxes3d.shape[0]
    noff = offsets.shape[0]

    xyz_offsets = boxes3d[:, None, 3:6] * offsets[None, :, :]
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = corners3d.reshape(nbox, -1, 1, 3) + xyz_offsets.reshape(1, 1, -1, 3)
    corners3d = common_utils.rotate_points_along_z(corners3d.reshape(nbox, -1, 3), boxes3d[:, 6]).view(nbox, noff, 8, 3)
    corners3d += boxes3d[:, None, None, 0:3]

    return corners3d
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


# cluster foreground / background?
def calc_angles(pts, curr_pt):
    curr_pt = curr_pt.reshape(1, -1)

    l1 = - curr_pt # to origin
    l2 = (pts - curr_pt)

    return torch.arccos((l1 @ l2.t()).reshape(-1) / (torch.norm(l1) * torch.norm(l2, dim=1) + 1e-8))

class FrustumClusterProposer(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
            lq: float = 0.33, uq: float = 0.35, cq: float = 0.5, \
            iou_w=0.9, dst_w=0.1, dns_w=0.5, \
            min_cam_iou=0.1,  \
            min_dist=1.0, max_dist=60, \
            score_thr=0.1, num_mags=10, min_cluster_size=10, \
            topk=1, angle_thr=torch.pi/2,  \
            cluster_method: ClusterMethod = ClusterMethod.HDB.value,
            nms_2d: float = 0.4, 
            nms_3d: float = 0.7
    ):
        super(FrustumClusterProposer, self).__init__()

        if model_cfg is not None:
            params_dict = model_cfg.PARAMS

            lq = params_dict.get('lq', lq)
            uq = params_dict.get('uq', uq)
            cq = params_dict.get('cq', cq)

            iou_w = params_dict.get('iou_w', iou_w)
            dst_w = params_dict.get('dst_w', dst_w)
            dns_w = params_dict.get('dns_w', dns_w)

            min_cam_iou = params_dict.get('min_cam_iou', min_cam_iou) 

            score_thr = params_dict.get('score_thr', score_thr)
            angle_thr = params_dict.get('angle_thr', angle_thr)
            num_mags = params_dict.get('num_mags', num_mags)
            min_cluster_size = params_dict.get('min_cluster_size', min_cluster_size)
            min_dist = params_dict.get('min_dist', min_dist)
            max_dist = params_dict.get('max_dist', max_dist)
            cluster_method = params_dict.get('cluster_method', cluster_method)
            nms_2d = params_dict.get('nms_2d', nms_2d)
            nms_3d = params_dict.get('nms_3d', nms_3d)
            topk = params_dict.get('topk', topk)

        self.angle_thr = angle_thr
        self.class_names = class_names
        self.class_labels = None

        if class_names is not None:
            self.class_labels = [i for i, x in enumerate(all_class_names) if x in class_names]
        else:
            raise TypeError('class_names should not be none!')

        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [900, 1600]

        self.hdbcluster = HDBSCANCluster(min_cluster_size=min_cluster_size)
        # self.anglecluster = AngleClusterer()
        self.combanglecluster = CombinedAngleClusterer(self.angle_thr)

        self.topk = topk

        # how many of each box variation we will include
        self.num_mags = num_mags

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.score_thr = score_thr

        self.iou_w = iou_w
        self.dst_w = dst_w
        self.dns_w = dns_w

        self.min_cam_iou = min_cam_iou
        self.max_clusters = 20
        self.cluster_method = cluster_method
        self.nms_2d = nms_2d
        self.nms_3d = nms_3d

        self.lq = lq
        self.uq = uq
        self.cq = cq

        num_proposals = self.num_mags
        print(f'generating {num_proposals} proposals')

        self.frustum_min = torch.tensor(self.min_dist, device='cuda')
        self.frustum_max = torch.tensor(self.max_dist, device='cuda')

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

    def create_box_proposals_og(self, curr_anchor: Tensor, curr_rotations: Tensor):
        base_boxes = torch.zeros((len(curr_rotations), 7), device='cuda')
        base_boxes[..., [3, 4, 5]] = curr_anchor
        base_boxes[..., 6] = curr_rotations

        base_boxes = base_boxes.reshape(-1, 7)
        base_corners = boxes_to_corners_3d(base_boxes).reshape(-1, 8, 3)

        return base_boxes, base_corners
    
    def create_box_proposals_offsets(self, curr_anchor: Tensor, curr_rotations: Tensor, centre: Tensor):
        curr_box_proposals = torch.zeros((len(curr_rotations), len(self.offset_vals_tensor), 7), device='cuda')

        curr_box_proposals[:, :, [3, 4, 5]] = curr_anchor
        for j, offset in enumerate(self.offset_vals_tensor):
            for k, ry in enumerate(curr_rotations):
                xyz = 0.5 * offset * curr_anchor

                xyz = common_utils.rotate_points_along_z(xyz.reshape(1, 1, 3), ry.reshape(1)).reshape(3)
                # xyz = xyz + centre

                curr_box_proposals[k, j, 0:3] = xyz
                curr_box_proposals[k, j, -1] = ry

        curr_box_proposals = curr_box_proposals.reshape(-1, 7)
        corners = boxes_to_corners_3d(curr_box_proposals).reshape(-1, 8, 3)

        return curr_box_proposals, corners

    def calc_iou(self, batch_dict: Dict, c: int, b: int, cam_box: Tensor, curr_corner_proposals: Tensor):
        image_pos, _ = self.project_to_camera(batch_dict, curr_corner_proposals.reshape(-1, 3), b, c)
        image_pos = image_pos[..., :2].reshape(-1, 8, 2)

        # clamp to image dimensions
        image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
        image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

        # print('image_pos', image_pos.shape)
        # get bbox
        xy1 = image_pos.min(dim=1).values
        xy2 = image_pos.max(dim=1).values
        proj_boxes = torch.zeros((xy1.shape[0], 4))
        proj_boxes[:, 0:2] = xy1
        proj_boxes[:, 2:] = xy2

        # proj_boxes = torch.zeros((image_pos.shape[0], 4))

        # for i in range(image_pos.shape[0]):
        #     coords2d = [[x[0].item(), x[1].item()] for x in image_pos[i]]
        #     # print('coords2d', coords2d)
        #     box = post_process_coords(coords2d, imsize=(1600, 1600))

        #     if box is not None:
        #         # box is on the camera image
        #         x1, y1, x2, y2 = box 
        #         proj_boxes[i, 0] = x1
        #         proj_boxes[i, 1] = y1
        #         proj_boxes[i, 2] = x2
        #         proj_boxes[i, 3] = y2



        # calculate ious
        ious = box_iou(proj_boxes, cam_box.cpu().reshape(1, -1)).reshape(-1)
        # if proj_boxes.sum() > 0:
        #     print('proj_boxes', proj_boxes)
        #     print('cam_box', cam_box)
        #     print('ious', ious.max())
        return ious

    def get_frust_proposals_og_multifrust(self, cam_scores, cs, b, boxes, batch_dict, cluster_mean, cluster_points, base_boxes, base_corners):
        bev_pts_xyz = cluster_mean.reshape(1, 3) #+ torch.randn((self.num_mags, 3), device='cuda')

        # curr_corner_proposals = self.base_corners[label.item()].clone()
        curr_corner_proposals = base_corners.clone().unsqueeze(0)
        curr_corner_proposals = curr_corner_proposals + bev_pts_xyz[:, None, None, :]
        
        # box version of the corners -> should be [N, 7]
        # curr_box_proposals = self.base_boxes[label.item()].clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1) # [M, N, 7] M = bev_pts_xyz.shape[0]
        curr_box_proposals = base_boxes.clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1)
        curr_box_proposals[..., 0:3] = curr_box_proposals[..., 0:3] + bev_pts_xyz[:, None, :]
        
        curr_corner_proposals = curr_corner_proposals.reshape(-1, 8, 3)
        curr_box_proposals = curr_box_proposals.reshape(-1, 7)

        # face centres
        face_pts = torch.cat([curr_corner_proposals[..., face_idx, :].mean(dim=1, keepdim=True) for face_idx in BOX_FACE_IDX], dim=1)

        # now move back so the front of the box is now where the centre was
        closest_ranking = torch.softmax(- curr_corner_proposals.clone().norm(dim=2), dim=1) # softmin
        weighted_front_centres = (closest_ranking.reshape(-1, 8, 1) * curr_corner_proposals).sum(dim=1)

        # now move back so the front of the box is now where the centre was
        # closest_ranking = torch.softmax(- face_pts.norm(dim=2), dim=1) # softmin
        # weighted_front_centres = (closest_ranking.reshape(-1, 4, 1) * curr_corner_proposals).sum(dim=1)
        # face_norm = face_pts.norm(dim=2)
        # closest_idx = face_norm.min(dim=1).indices
        # weighted_front_centres = torch.cat([face_pts[i, idx].reshape(1, 3) for i, idx in enumerate(closest_idx)], dim=0)

        if curr_box_proposals.shape[0] != weighted_front_centres.shape[0]:
            print('face_pts', face_pts.shape, weighted_front_centres.shape, curr_box_proposals.shape, curr_corner_proposals.shape)

            print("box proposals dim0 != front_centres dim0")

        front_to_centre = (curr_box_proposals[..., 0:3] - weighted_front_centres)
        curr_box_proposals[..., 0:3] += front_to_centre
        curr_corner_proposals += front_to_centre.reshape(-1, 1, 3)

        # filter those too far away
        dist_to_origin = weighted_front_centres.norm(dim=-1)
        valid_proposals = dist_to_origin < self.max_dist
        
        if valid_proposals.sum() == 0:
            return None, None

        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        weighted_front_centres = weighted_front_centres[valid_proposals]

        # takes multicamera views into account
        ious_list = []
        for (c, box) in zip(cs, boxes):
            ious = self.calc_iou(batch_dict, c, b, box, curr_corner_proposals)
            ious_list.append(ious.unsqueeze(0))

        ious_per_box = torch.cat(ious_list, dim=0)

        # weighted based on cam score
        ious = (ious_per_box * cam_scores.reshape(-1, 1)).sum(dim=0) / cam_scores.sum()


        # print('ious_per_box', ious_per_box.shape, ious_per_box)
        # print('weighted ious', ious)
        # ious = ious_per_box.max(dim=0).values
        # ious = ious_per_box.max(dim=0).values

        if cluster_mean is not None:
            dists = torch.cdist(weighted_front_centres.reshape(-1, 3), cluster_mean.reshape(1, 3)).reshape(-1)
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
        ious = ious[valid_proposals]

        if ious.shape[0] == 0:
            return None, None

        # project image points back to lidar
        # box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
        #     camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
        #     post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        # )
        
        # calculate how many in each proposal
        num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
        if cluster_points.shape[0] > 0:
            point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cluster_points.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
            point_box_indices = point_box_indices.reshape(-1)
            
            for i in range(curr_box_proposals.shape[0]):
                num_pts_in_boxes[i] = (point_box_indices == i).sum()

        # soft_densities = (num_pts_in_boxes) / (num_pts_in_boxes.max() + 1e-8)
        soft_densities = torch.softmax(num_pts_in_boxes, dim=0)

        second_stage_scores = ((1.0 - self.dns_w) + soft_densities * self.dns_w) * ((1.0 - self.iou_w) + ious * self.iou_w) * ((1.0 - self.dst_w) + dists_ranked.cpu() * self.dst_w)

        # NMS and ordering for topk
        selected, _ = iou3d_nms_utils.nms_normal_gpu(curr_box_proposals, second_stage_scores, thresh=self.nms_3d)
        curr_box_proposals = curr_box_proposals[selected].contiguous()
        curr_corner_proposals = curr_corner_proposals[selected]
        ious = ious[selected]
        second_stage_scores = second_stage_scores[selected]
        num_pts_in_boxes = num_pts_in_boxes[selected]

        # filter topk proposals (already in order from NMS)
        valid_proposals = slice(min(self.topk, second_stage_scores.shape[0]))
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        ious = ious[valid_proposals]
        second_stage_scores = second_stage_scores[valid_proposals]
        num_pts_in_boxes = num_pts_in_boxes[valid_proposals]

        return curr_box_proposals, ious

    # (frust_box, c, box, box_points, weighted_centre_xyz, label, score)
    def get_frust_proposals_og(self, frust_box, label, c, b, box, batch_dict, weighted_centre_xyz, box_points_xyz, base_boxes, base_corners):
        frust_bev_box = torch.cat([frust_box[[2*i, 2*i+1], :].mean(dim=0)[None, :] for i in range(4)], dim=0)

        # magnitude of interpolating between closest frustum point to furtherest (producing boxes at different depths)
        mags = torch.linspace(0.0, 1.0, self.num_mags, device='cuda').reshape(-1, 1).repeat(1, 3)

        frust_center_close = frust_bev_box[:2, :].mean(dim=0)
        frust_center_far = frust_bev_box[2:, :].mean(dim=0)

        center_vec = (frust_center_far - frust_center_close)
        bev_pts_xyz = frust_center_close.reshape(1, 3) + center_vec * mags

        # curr_corner_proposals = self.base_corners[label.item()].clone()
        curr_corner_proposals = base_corners.clone().unsqueeze(0)
        curr_corner_proposals = curr_corner_proposals + bev_pts_xyz[:, None, None, :]
        
        # box version of the corners -> should be [N, 7]
        # curr_box_proposals = self.base_boxes[label.item()].clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1) # [M, N, 7] M = bev_pts_xyz.shape[0]
        curr_box_proposals = base_boxes.clone().unsqueeze(0).repeat(bev_pts_xyz.shape[0], 1, 1)
        curr_box_proposals[..., 0:3] = curr_box_proposals[..., 0:3] + bev_pts_xyz[:, None, :]
        
        curr_corner_proposals = curr_corner_proposals.reshape(-1, 8, 3)
        curr_box_proposals = curr_box_proposals.reshape(-1, 7)

        # face centres
        face_pts = torch.cat([curr_corner_proposals[..., face_idx, :].mean(dim=1, keepdim=True) for face_idx in BOX_FACE_IDX], dim=1)

        # now move back so the front of the box is now where the centre was
        closest_ranking = torch.softmax(- curr_corner_proposals.clone().norm(dim=2), dim=1) # softmin
        weighted_front_centres = (closest_ranking.reshape(-1, 8, 1) * curr_corner_proposals).sum(dim=1)

        # now move back so the front of the box is now where the centre was
        # closest_ranking = torch.softmax(- face_pts.norm(dim=2), dim=1) # softmin
        # weighted_front_centres = (closest_ranking.reshape(-1, 4, 1) * curr_corner_proposals).sum(dim=1)
        # face_norm = face_pts.norm(dim=2)
        # closest_idx = face_norm.min(dim=1).indices
        # weighted_front_centres = torch.cat([face_pts[i, idx].reshape(1, 3) for i, idx in enumerate(closest_idx)], dim=0)

        if curr_box_proposals.shape[0] != weighted_front_centres.shape[0]:
            print('face_pts', face_pts.shape, weighted_front_centres.shape, curr_box_proposals.shape, curr_corner_proposals.shape)

            print("box proposals dim0 != front_centres dim0")

        front_to_centre = (curr_box_proposals[..., 0:3] - weighted_front_centres)
        curr_box_proposals[..., 0:3] += front_to_centre
        curr_corner_proposals += front_to_centre.reshape(-1, 1, 3)

        # filter those too far away
        dist_to_origin = weighted_front_centres.norm(dim=-1)
        valid_proposals = dist_to_origin < self.max_dist
        
        if valid_proposals.sum() == 0:
            return None, None

        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        weighted_front_centres = weighted_front_centres[valid_proposals]

        # project all the proposed boxes to the image (so we can calculate the IOU with 2D prediction)
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
            return None, None

        # project image points back to lidar
        # box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
        #     camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
        #     post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        # )
        
        # calculate how many in each proposal
        num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
        if box_points_xyz.shape[0] > 0:
            point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(box_points_xyz.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
            point_box_indices = point_box_indices.reshape(-1)
            
            for i in range(curr_box_proposals.shape[0]):
                num_pts_in_boxes[i] = (point_box_indices == i).sum()

        # soft_densities = (num_pts_in_boxes) / (num_pts_in_boxes.max() + 1e-8)
        soft_densities = torch.softmax(num_pts_in_boxes, dim=0)

        second_stage_scores = ((1.0 - self.dns_w) + soft_densities * self.dns_w) * ((1.0 - self.iou_w) + ious * self.iou_w) * ((1.0 - self.dst_w) + dists_ranked.cpu() * self.dst_w)

        # NMS and ordering for topk
        selected, _ = iou3d_nms_utils.nms_normal_gpu(curr_box_proposals, second_stage_scores, thresh=self.nms_3d)
        curr_box_proposals = curr_box_proposals[selected].contiguous()
        curr_corner_proposals = curr_corner_proposals[selected]
        ious = ious[selected]
        second_stage_scores = second_stage_scores[selected]
        num_pts_in_boxes = num_pts_in_boxes[selected]

        # filter topk proposals (already in order from NMS)
        valid_proposals = slice(min(self.topk, second_stage_scores.shape[0]))
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        ious = ious[valid_proposals]
        second_stage_scores = second_stage_scores[valid_proposals]
        num_pts_in_boxes = num_pts_in_boxes[valid_proposals]

        return curr_box_proposals, ious

    def get_proposals(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        batch_boxes = []
        batch_labels = []
        batch_scores = []
        batch_idx = []
        
        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            cur_points_dirs = cur_points.clone()#.unsqueeze(0)
            cur_points_mags = torch.norm(cur_points_dirs, dim=-1, keepdim=True)
            cur_points_dirs = cur_points_dirs / cur_points_mags

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            scene_frusts = []
            scene_cam_boxes = []
            scene_frusts_cams = []
            scene_frusts_points = []
            scene_weighted_centres = []
            scene_scores = []
            scene_labels = []

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                # 2d nms
                if cam_boxes.shape[0] > 0 and self.nms_2d > 0.05:
                    selected = batched_nms(cam_boxes, cam_scores, cam_labels, self.nms_2d)
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


                    on_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                        & (cam_points[..., 2] >= self.frustum_min)
                        & (cam_points[..., 2] <= self.frustum_max)
                    )

                    box_points = cam_points[on_box]

                    if box_points.numel() > 0:
                        cur_frustum_min = torch.quantile(box_points[:, 2], self.lq)
                        cur_frustum_max = torch.quantile(box_points[:, 2], self.uq)
                        # weighted_centre_cam = dists_ranking @ box_points
                        cur_centre_z = torch.quantile(box_points[:, 2], self.cq)
                        weighted_centre_cam = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2, cur_centre_z[None].cpu()]).reshape(1, -1).cuda()

                        weighted_centre_xyz = self.get_geometry_at_image_coords(weighted_centre_cam.reshape(-1, 3), [c], [b], 
                            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                        )
                    else:
                        continue

                    # bev_mask = self.get_frustum_bev_mask(frust_box)
                    box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    cur_frustum_max = torch.minimum(cur_frustum_max, self.frustum_max)
                    cur_frustum_min = torch.maximum(cur_frustum_min, self.frustum_min)

                    box = box.cuda()
                    xyzxyz = torch.cat([box[0][None], box[1][None], cur_frustum_min[None], box[2][None], box[3][None], cur_frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    scene_frusts.append(frust_box)

                    scene_cam_boxes.append(box)
                    scene_frusts_cams.append(c)
                    scene_frusts_points.append(box_points_xyz)
                    scene_weighted_centres.append(weighted_centre_xyz)

                    scene_scores.append(score[None])
                    scene_labels.append(label[None])

            # nothing to do
            if len(scene_frusts_points) == 0:
                continue

            # cluster together!
            all_box_points_idx = torch.cat([torch.tensor([i] * len(x)) for i, x in enumerate(scene_frusts_points)])
            all_box_points_labels = torch.cat([torch.tensor([scene_labels[i].item()] * len(x)) for i, x in enumerate(scene_frusts_points)])
            
            all_box_points = torch.cat(scene_frusts_points)
            all_box_points_labels = all_box_points_labels.to(all_box_points.device)

            if self.cluster_method == ClusterMethod.HDB.value:
                all_point_feats = torch.cat((all_box_points, all_box_points_labels.reshape(-1, 1)), dim=1)
                all_cluster_labels = self.hdbcluster(all_point_feats, self.anchors)
            elif self.cluster_method == ClusterMethod.ANGLE.value:
                all_cluster_labels = self.combanglecluster(all_box_points, all_box_points_labels, self.anchors)
            else:
                raise ValueError(f'cluster_method={self.cluster_method} is not valid')
            
            num_clusters = all_cluster_labels.max() + 1

            # print('num per cluster', [(all_cluster_labels == i).sum().item() for i in range(num_clusters)])

            # cluster join
            # cluster_centres = []
            # cluster_labels = []
            # cluster_n_pts = []
            # cluster_rs = []
            # for i in range(num_clusters):
            #     cluster_mask = (all_cluster_labels == i)
            #     cluster_points = all_box_points[cluster_mask]

            #     if cluster_points.shape[0] < 10:
            #         continue

            #     curr_label = all_box_points_labels[cluster_mask][0]
            #     curr_anchor = self.anchors[curr_label.item()]
                
            #     cluster_mean = cluster_points.mean(dim=0, keepdim=True)

            #     cluster_centres.append(cluster_mean)
            #     cluster_labels.append(cluster_labels)
            #     cluster_n_pts.append(cluster_points.shape[0])
            #     cluster_rs.append(curr_anchor.norm() / 2.0)

            # cluster_n_pts = torch.tensor(cluster_n_pts)
            # order = torch.argsort(-cluster_n_pts)
            # keep = torch.ones_like(order, dtype=torch.bool)

            # for i in range(num_clusters):
            #     curr_idx = order[i]
            #     curr_centre = cluster_centres[curr_idx]
            #     curr_lbl = cluster_labels[curr_idx]

            #     if not keep[i]:
            #         continue

            #     for j in range(i + 1, num_clusters):
            #         oth_idx = order[j]
            #         oth_centre = order[oth_idx]
            #         oth_r = cluster_rs[oth_idx]
            #         oth_lbl = cluster_labels[oth_idx]

            #         dist = (curr_centre - oth_centre).norm()

            #         if curr_lbl == oth_lbl and dist < oth_r:
            #             keep[j] = False

            #             # relabel
            #             oth_mask = (all_cluster_labels == oth_idx)
            #             all_cluster_labels[oth_mask] = curr_idx

            # print('clusters removed', (keep.numel() - keep.sum()))
            # print('num per cluster', [(all_cluster_labels == i).sum().item() for i in range(num_clusters)])


            all_proposals = []
            all_proposals_idx = []
            all_proposals_labels = []
            all_proposal_scores = []
            all_proposal_cam_scores = []

            for i in range(num_clusters):
                cluster_mask = (all_cluster_labels == i)
                cluster_points = all_box_points[cluster_mask]

                if cluster_points.shape[0] < 10:
                    continue

                cluster_labels = all_box_points_labels[cluster_mask]
                cluster_frustum_idx = all_box_points_idx[cluster_mask]
                frust_idx_set = set(x.item() for x in cluster_frustum_idx)

                curr_label = cluster_labels[0].item()
                curr_anchor = self.anchors[curr_label]
                
                cluster_mean = cluster_points.mean(dim=0, keepdim=True)
                rel_points = (cluster_points.clone() - cluster_mean)

                # svd
                (U, S, Vh) = torch.linalg.svd(rel_points)

                dirf = (S.reshape(-1, 1) * Vh).sum(dim=0)
                dirf = dirf / dirf.norm(dim=-1, keepdim=True)

                dirs = torch.cat((Vh, dirf.reshape(1, 3)), dim=0)

                curr_rotations = torch.atan2(dirs[..., 1], dirs[..., 0])
                curr_rotations = torch.cat((curr_rotations, curr_rotations + torch.pi/2))

                # curr_rotations = torch.linspace(-torch.pi/2, torch.pi/2, 10, device='cuda')

                base_boxes, base_corners = self.create_box_proposals_og(curr_anchor, curr_rotations)
                # base_boxes, base_corners = self.create_box_proposals_offsets(curr_anchor, curr_rotations, cluster_mean)

                # cs = []
                # boxes = []
                # cam_scores = []
                # for frust_idx in frust_idx_set:
                    # cs.append(scene_frusts_cams[frust_idx])
                    # boxes.append(scene_cam_boxes[frust_idx])
                    # cam_scores.append(scene_scores[frust_idx].item())
                # cs = [scene_frusts_cams[frust_idx] for frust_idx in frust_idx_set]
                # boxes = [scene_cam_boxes[frust_idx] for frust_idx in frust_idx_set]

                cam_scores = torch.tensor([scene_scores[frust_idx].item() for frust_idx in frust_idx_set if scene_labels[frust_idx].item() == curr_label])
                cam_score = cam_scores.mean().item()
                cs = [scene_frusts_cams[frust_idx] for frust_idx in frust_idx_set if scene_labels[frust_idx] == curr_label]
                boxes = [scene_cam_boxes[frust_idx] for frust_idx in frust_idx_set if scene_labels[frust_idx] == curr_label]

    
                proposal_boxes, proposal_score = self.get_frust_proposals_og_multifrust(cam_scores, cs, b, boxes, batch_dict, cluster_mean, cluster_points, base_boxes, base_corners)
                if proposal_boxes is not None:
                    for box, score in zip(proposal_boxes, proposal_score):
                        all_proposals.append(box)
                        all_proposal_scores.append(score.item())
                        all_proposal_cam_scores.append(cam_score)
                        all_proposals_idx.append(0) # whatever
                        all_proposals_labels.append(int(curr_label))
                
                # per frustum
                # for frust_idx in frust_idx_set:
                #     frust_box = scene_frusts[frust_idx]
                #     label = scene_labels[frust_idx]
                #     c = scene_frusts_cams[frust_idx]
                #     box = scene_cam_boxes[frust_idx]
                #     cam_score = scene_scores[frust_idx]
                #     weighted_centre_xyz = scene_weighted_centres[frust_idx]

                #     proposal_boxes, proposal_score = self.get_frust_proposals_og(frust_box, label, c, b, box, batch_dict, weighted_centre_xyz, cluster_points, base_boxes, base_corners)

                #     if proposal_boxes is not None:
                #         for box, score in zip(proposal_boxes, proposal_score):
                #             all_proposals.append(box)
                #             all_proposal_scores.append(score.item())
                #             all_proposal_cam_scores.append(cam_score)
                #             all_proposals_idx.append(frust_idx)
                #             all_proposals_labels.append(int(curr_label))

            if len(all_proposals) == 0:
                continue

            # concatenate all frustum proposals
            all_proposals = torch.stack(all_proposals, dim=0)
            all_scores = torch.tensor(all_proposal_scores)
            all_cam_scores = torch.tensor(all_proposal_cam_scores)
            all_proposals_labels = torch.tensor(all_proposals_labels)
            # all_proposals_idx = torch.tensor(all_proposals_idx)
            
            # (iou + cls) / 2
            balanced_scores = (all_scores + all_cam_scores) / 2.0

            # nms for this batch
            selected, _ = iou3d_nms_utils.nms_normal_gpu(all_proposals, balanced_scores, thresh=self.nms_3d)
            # selected = batched_distance_nms(all_proposals, balanced_scores, all_proposals_labels)
            all_proposals = all_proposals[selected]
            all_proposals_labels = all_proposals_labels[selected]
            # all_scores = all_scores[selected]
            # all_cam_scores = all_cam_scores[selected]
            balanced_scores = balanced_scores[selected]

            # add to all batch
            for box, score, label in zip(all_proposals, balanced_scores, all_proposals_labels):
                batch_boxes.append(box)
                batch_scores.append(score)
                batch_labels.append(label)
                batch_idx.append(b)

        # return output
        if len(batch_boxes) > 0:
            batch_boxes = torch.cat(batch_boxes, dim=0).reshape(-1, 7)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            batch_scores = torch.tensor(batch_scores)
            batch_idx = torch.tensor(batch_idx)
        else:
            batch_boxes = torch.zeros((0, 7), device='cuda')
            batch_labels = torch.zeros((0), dtype=torch.long)
            batch_scores = torch.zeros((0))
            batch_idx = torch.zeros((0), dtype=torch.long)

        return batch_boxes, batch_labels, batch_scores, batch_idx

    def points_in_boxes(self, boxes3d, points): # very expensive (do not put all pts in here)!
        N = boxes3d.shape[0]

        points = points.clone()
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        # instead of rotating the box, rotate the pointcloud (so can simply threshold the points)
        points = points[:, None, :] - boxes3d[None, :, 0:3] # centre the points
        points = points.permute(1, 0, 2)

        points = common_utils.rotate_points_along_z(points, - boxes3d[:, 6]).view(N, -1, 3) # rotate by negative angle

        in_box = (
            (points[..., 0] >= corners3d[..., 0].min(dim=1, keepdim=True).values) &
            (points[..., 0] <= corners3d[..., 0].max(dim=1, keepdim=True).values) &
            
            (points[..., 1] >= corners3d[..., 1].min(dim=1, keepdim=True).values) &
            (points[..., 1] <= corners3d[..., 1].max(dim=1, keepdim=True).values) &
            
            (points[..., 2] >= corners3d[..., 2].min(dim=1, keepdim=True).values) &
            (points[..., 2] <= corners3d[..., 2].max(dim=1, keepdim=True).values)
            )

        return in_box, points

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()

        # print('current coords', cur_coords)
        # print('image_paths', batch_dict['image_paths'])

        # camera_intrinsics = batch_dict['camera_intrinsics']
        # camera2lidar = batch_dict['camera2lidar']
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
            mask = (proposed_batch_idx == k)

            # ret_dict.append()
            # print('boxes', proposed_boxes[mask].shape)
            # print('scores', proposed_scores[mask].shape)
            # print('labels', proposed_labels[mask].shape)

            batch_boxes = proposed_boxes[mask]
            scene_scores = proposed_scores[mask]
            scene_labels = proposed_labels[mask]

            batch_boxes = batch_boxes.to(dtype=torch.float)
            scene_scores = scene_scores.to(dtype=torch.float)

            ret_dict[k]['pred_boxes'] = batch_boxes#.cpu()
            ret_dict[k]['pred_scores'] = scene_scores
            ret_dict[k]['pred_labels'] = scene_labels.int() + 1

        return ret_dict 