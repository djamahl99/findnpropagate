import argparse
import glob
from pathlib import Path

import random

import numpy as np
import torch
from torch import nn, Tensor

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

class HDBSCANCluster:
    def __init__(self):
        self.hdb = HDBSCAN(min_cluster_size=5)

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

            print('curr_angles', curr_angles.max(), curr_angles.min())

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

    def __call__(self, frust_points: Tensor, frust_labels: Tensor, anchors: Tensor, max_clusters: int):
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

        print('removed clusters', order[~keep])


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

class FrustumProposer(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
            iou_w=0.9, dns_w=0.5, occl_w=0.1, inlier_w=0.1, \
            min_cam_iou=0.1,  \
            ry_min=0.0, ry_max=torch.pi, \
            num_mags=10, min_dist=1.0, max_dist=60, \
            num_rot=10, score_thr=0.1, \
            topk=1, angle_thr=torch.pi/2, bg_thr=0.5
    ):
        super(FrustumProposer, self).__init__()

        if model_cfg is not None:
            params_dict = model_cfg.PARAMS

            iou_w = params_dict.get('iou_w', iou_w)
            dns_w = params_dict.get('dns_w', dns_w)
            occl_w = params_dict.get('occl_w', occl_w)
            inlier_w = params_dict.get('inlier_w', inlier_w)

            min_cam_iou = params_dict.get('min_cam_iou', min_cam_iou) 

            bg_thr = params_dict.get('bg_thr', bg_thr)
            score_thr = params_dict.get('score_thr', score_thr)
            angle_thr = params_dict.get('angle_thr', angle_thr)
            num_mags = params_dict.get('num_mags', num_mags)
            min_dist = params_dict.get('min_dist', min_dist)
            max_dist = params_dict.get('max_dist', max_dist)
            num_rot = params_dict.get('num_rot', num_rot)
            topk = params_dict.get('topk', topk)

        self.angle_thr = angle_thr
        self.class_names = class_names
        self.class_labels = None
        if class_names is not None:
            self.class_labels = [i for i, x in enumerate(all_class_names) if x in class_names]

        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [900, 1600]

        self.hdbcluster = HDBSCANCluster()
        # self.anglecluster = AngleClusterer()
        self.combanglecluster = CombinedAngleClusterer(self.angle_thr)

        self.topk = topk

        self.ry_min = ry_min
        self.ry_max = ry_max

        # how many of each box variation we will include
        self.num_mags = num_mags
        self.num_rot = num_rot

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.score_thr = score_thr
        self.bg_thr = bg_thr

        self.iou_w = iou_w
        self.occl_w = occl_w
        self.dns_w = dns_w
        self.inlier_w = inlier_w

        self.min_cam_iou = min_cam_iou
        self.max_clusters = 20


        num_proposals = self.num_mags * self.num_rot
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
            # (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            # (0, 0, -1),
            (0, -1, 0),
            (-1, 0, 0),
            (0, 0, 0)
        ]
        self.offset_vals_tensor = torch.tensor(offset_vals, device='cuda')

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

        # calculate ious
        ious = box_iou(proj_boxes, cam_box.cpu().reshape(1, -1)).reshape(-1)
        
        return ious

    def get_frust_proposals_multifrust(self, curr_anchor, curr_box_proposals, cs, b, boxes, batch_dict, cur_points, cur_points_dirs, cur_points_mags):
        curr_box_proposals = curr_box_proposals.reshape(-1, 7)
        curr_corner_proposals = boxes_to_corners_3d(curr_box_proposals).reshape(-1, 8, 3)

        # takes multicamera views into account
        ious_list = []
        for (c, box) in zip(cs, boxes):
            ious = self.calc_iou(batch_dict, c, b, box, curr_corner_proposals)
            ious_list.append(ious.unsqueeze(0))

        ious_per_box = torch.cat(ious_list, dim=0)

        # print('ious_per_box', ious_per_box.shape, ious_per_box.max(dim=0).values)
        ious = ious_per_box.max(dim=0).values
        # print('ious', ious.max())

        # reject under iou lower bound
        # valid_proposals = ious >= torch.maximum(self.min_cam_iou, ious.mean())
        valid_proposals = ious >= self.min_cam_iou
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        ious = ious[valid_proposals]
        # dists_ranked = dists_ranked[valid_proposals]

        if ious.shape[0] == 0:
            return None, None

        occlusion_scores = self.calc_occl_scores(curr_anchor, curr_box_proposals, cur_points, cur_points_dirs, cur_points_mags).cpu()
        inlier_scores = self.calc_inlier_scores(curr_anchor, curr_box_proposals, cur_points, cur_points_dirs, cur_points_mags).cpu()

        num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
        point_box_indices = point_box_indices.reshape(-1)
        
        for i in range(curr_box_proposals.shape[0]):
            num_pts_in_boxes[i] = (point_box_indices == i).sum()

        occlusion_scores_ranked = torch.softmax(-occlusion_scores, dim=0)
        inlier_scores_ranked = torch.softmax(-inlier_scores, dim=0)
        soft_densities = torch.softmax(num_pts_in_boxes, dim=0)

        second_stage_scores = ious * self.iou_w + inlier_scores_ranked * self.inlier_w + soft_densities * self.dns_w + occlusion_scores_ranked * self.occl_w

        if self.topk > 1:
            valid_proposals = torch.topk(second_stage_scores, k=min(self.topk, second_stage_scores.shape[0])).indices
        else:
            valid_proposals = torch.argmax(second_stage_scores).reshape(-1)
        curr_corner_proposals = curr_corner_proposals[valid_proposals]
        curr_box_proposals = curr_box_proposals[valid_proposals]
        second_stage_scores = second_stage_scores[valid_proposals]
        ious = ious[valid_proposals]
        num_pts_in_boxes = num_pts_in_boxes[valid_proposals]

        occlusion_scores = -occlusion_scores[valid_proposals]
        inlier_scores = -inlier_scores[valid_proposals]

        # out_scores = torch.sigmoid(-occlusion_scores[valid_proposals]) + ious

        return curr_box_proposals, num_pts_in_boxes

    def calc_occl_scores(self, anchor: Tensor, box_proposals: Tensor, points: Tensor, dirs: Tensor, mags: Tensor):
        box_proposals = box_proposals.reshape(1, -1, 7)
        nboxes = box_proposals.shape[1]

        occl_scores = points.new_zeros((nboxes,))

        phi = anchor.min() / 2.0
        mags_empty = mags - phi
        mags_occl = mags + phi

        # generate query points
        empty_points = dirs * mags_empty
        occl_points = dirs * mags_occl

        # real points score
        real_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(points.reshape(1, -1, 3), box_proposals)

        # empty points
        empty_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(empty_points.reshape(1, -1, 3), box_proposals)

        # occl points
        occl_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(occl_points.reshape(1, -1, 3), box_proposals)

        for i in range(nboxes):
            real_mask = (real_point_box_indices == i).reshape(-1)
            num_real = real_mask.sum()
            
            empty_mask = (empty_point_box_indices == i).reshape(-1)
            num_empty = empty_mask.sum()

            occl_mask = (occl_point_box_indices == i).reshape(-1)
            num_occl = occl_mask.sum()
            
            occl_scores[i] = (num_occl + num_empty - 2*num_real) / (points.shape[0]*2)
        
        return occl_scores

    def calc_inlier_scores(self, anchor: Tensor, box_proposals: Tensor, points: Tensor, dirs: Tensor, mags: Tensor):
        coeffs3 = points.new_tensor([[0, 0, 1]])

        a0, a1, a2 = anchor/2.0
        
        def cost_fn(centre: Tensor, coeffs1: Tensor, coeffs2: Tensor):
            coeffs1 = coeffs1.reshape(1, 3)
            coeffs2 = coeffs2.reshape(1, 3)
            centre = centre.reshape(1, 3)

            inlier_points = points
            centered_points = (inlier_points - centre)

            points_proj1 = centered_points * coeffs1
            points_proj1 = points_proj1.sum(dim=-1)

            points_proj2 = centered_points * coeffs2
            points_proj2 = points_proj2.sum(dim=-1)

            points_proj3 = centered_points * coeffs3
            points_proj3 = points_proj3.sum(dim=-1)

            # return points_proj1.pow(2).mean() + points_proj2.pow(2).mean() + points_proj3.pow(2).mean()
            dists_anchor0 = torch.relu(points_proj1.abs() - a0) #/ a0
            dists_anchor1 = torch.relu(points_proj2.abs() - a1) #/ a1
            dists_anchor2 = torch.relu(points_proj3.abs() - a2) #/ a2

            return dists_anchor0.pow(2).mean() + dists_anchor1.pow(2).mean() + dists_anchor2.pow(2).mean()


        nboxes = box_proposals.shape[0]
        scores = box_proposals.new_zeros((nboxes,)) 

        centres = box_proposals[..., 0:3]

        for i in range(nboxes):
            ry = box_proposals[i, 6]
            coeffs1 = torch.stack([torch.cos(ry), torch.sin(ry), torch.cos(ry)*0.0])
            coeffs2 = torch.stack([torch.cos(ry + torch.pi/2), torch.sin(ry + torch.pi/2),  torch.cos(ry)*0.0])

            scores[i] = cost_fn(centres[i], coeffs1, coeffs2)

        return scores

    def create_box_proposals_(self, curr_anchor: Tensor, curr_rotations: Tensor, centre: Tensor):
        curr_box_proposals = torch.zeros((len(curr_rotations), len(self.offset_vals_tensor), 7), device='cuda')

        curr_box_proposals[:, :, [3, 4, 5]] = curr_anchor
        for j, offset in enumerate(self.offset_vals_tensor):
            for k, ry in enumerate(curr_rotations):
                xyz = 0.5 * offset * curr_anchor

                xyz = common_utils.rotate_points_along_z(xyz.reshape(1, 1, 3), ry.reshape(1)).reshape(3)
                xyz = xyz + centre

                curr_box_proposals[k, j, 0:3] = xyz
                curr_box_proposals[k, j, -1] = ry

        return curr_box_proposals

    def create_box_proposals(self, curr_anchor: Tensor, curr_rotations: Tensor, geo_min: Tensor, geo_max: Tensor):
        # curr_box_proposals = torch.zeros((self.num_mags, len(self.offset_vals_tensor), self.num_rot, 7), device='cuda')
        curr_box_proposals = torch.zeros((self.num_rot, self.num_mags*3, 7), device='cuda')

        orthog_matrix = torch.tensor([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], device=geo_min.device, dtype=geo_min.dtype)

        geo_vec = (geo_max - geo_min)
        geo_vec_dir = geo_vec / geo_vec.norm(dim=-1, keepdim=True)

        orthog_vec = geo_vec_dir @ orthog_matrix

        rs = torch.linspace(0, 1, self.num_mags, device='cuda')
        centres = geo_min.reshape(1, 3) + geo_vec.reshape(1, 3) * rs.reshape(-1, 1)

        a1 = curr_anchor[:2].min() / 2.0
        a1_offset = (orthog_vec * a1).reshape(1, 3)

        centres1 = centres.reshape(-1, 3) + a1_offset 
        centres2 = centres.reshape(-1, 3) - a1_offset 
        centres = torch.cat((centres1, centres, centres2), dim=0)

        # fill empty box proposals
        curr_box_proposals[:, :, [3, 4, 5]] = curr_anchor
        curr_box_proposals[:, :, :3] = centres.reshape(1, -1, 3)
        curr_box_proposals[:, :, 6] = curr_rotations.reshape(-1, 1)

        return curr_box_proposals

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

        proposal_boxes = []
        frust_scores = []
        frust_batch_idx = []
        frust_labels = []

        
        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            cur_points_dirs = cur_points.clone()#.unsqueeze(0)
            cur_points_mags = torch.norm(cur_points_dirs, dim=-1, keepdim=True)
            cur_points_dirs = cur_points_dirs / cur_points_mags


            # print('plane_coeff', plane_coeff)

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            batch_cam_boxes = []
            batch_frusts_cams = []
            batch_frusts_points = []
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
                        & (cam_points[..., 2] >= self.frustum_min)
                        & (cam_points[..., 2] <= self.frustum_max)
                    )

                    box_points = cam_points[on_box]

                    if box_points.numel() == 0:
                        # print(f'no points for {label}!')
                        continue

                    # bev_mask = self.get_frustum_bev_mask(frust_box)
                    box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    batch_cam_boxes.append(box)
                    batch_frusts_cams.append(c)
                    batch_frusts_points.append(box_points_xyz)

                    batch_scores.append(score[None])
                    batch_labels.append(label[None])

            if len(batch_frusts_points) == 0:
                continue

            # cluster together!
            all_box_points_idx = torch.cat([torch.tensor([i] * len(x)) for i, x in enumerate(batch_frusts_points)])
            all_box_points_labels = torch.cat([torch.tensor([batch_labels[i].item()] * len(x)) for i, x in enumerate(batch_frusts_points)])

            all_box_points = torch.cat(batch_frusts_points)
            all_box_points_labels = all_box_points_labels.to(all_box_points.device)

            # label them
            # all_cluster_labels = self.combanglecluster(all_box_points, all_box_points_labels, self.anchors, max_clusters=max(100, len(batch_frusts_points)))
            # all_cluster_labels = self.combanglecluster(all_box_points, all_box_points_labels, self.anchors, max_clusters=max(100, len(batch_frusts_points)))
            all_point_feats = torch.cat((all_box_points, all_box_points_labels.reshape(-1, 1)), dim=1)
            all_cluster_labels = self.hdbcluster(all_point_feats, self.anchors)

            num_clusters = all_cluster_labels.max()

            all_proposals = []
            all_proposals_idx = []
            all_proposal_scores = []
            all_proposal_cam_scores = []

            for i in range(num_clusters + 1):
                cluster_mask = (all_cluster_labels == i)
                cluster_points = all_box_points[cluster_mask]
                
                cluster_points_mags = cluster_points.norm(dim=-1, keepdim=True)
                cluster_points_dirs = cluster_points / cluster_points_mags

                if cluster_points.shape[0] < 10:
                    continue

                cluster_labels = all_box_points_labels[cluster_mask]
                cluster_frustum_idx = all_box_points_idx[cluster_mask]
                frust_idx_set = set(x.item() for x in cluster_frustum_idx)

                curr_label = cluster_labels[0]
                frust_idx = cluster_frustum_idx[0]
                curr_anchor = self.anchors[curr_label.item()]
                curr_anchor_r = curr_anchor.norm() / 2.0

                cluster_mean = cluster_points.mean(dim=0, keepdim=True)
                rel_points = (cluster_points.clone() - cluster_mean)

                (U, S, Vh) = torch.linalg.svd(rel_points)

                if S.min() < self.bg_thr:
                    # probably background
                    continue


                dirf = (S.reshape(-1, 1) * Vh).sum(dim=0)
                dirf = dirf / dirf.norm(dim=-1, keepdim=True)

                geo_min = cluster_mean - dirf * curr_anchor_r
                geo_max = cluster_mean + dirf * curr_anchor_r

                curr_rotations = torch.linspace(-torch.pi/2, torch.pi/2, self.num_rot, device='cuda')
                curr_box_proposals = self.create_box_proposals(curr_anchor, curr_rotations, geo_min, geo_max)

                cs = [c for i, c in enumerate(batch_frusts_cams) if i in frust_idx_set]
                boxes = [x for i, x in enumerate(batch_cam_boxes) if i in frust_idx_set]
                cam_scores = [x for i, x in enumerate(batch_scores) if i in frust_idx_set]
                cam_score = max(cam_scores)
                # print('cams, boxes, cam_scores', cs, boxes, cam_scores, cam_score)

                box_proposals, proposal_score = self.get_frust_proposals_multifrust(curr_anchor, curr_box_proposals, cs, b, boxes, batch_dict, cluster_points, cluster_points_dirs, cluster_points_mags)
                # proposal_corners, proposal_score = self.get_frust_proposals_multifrust(curr_anchor, curr_box_proposals, cs, b, boxes, batch_dict, all_box_points, all_box_points_dirs, all_box_points_mags)

                if box_proposals is not None:
                    for box, score in zip(box_proposals, proposal_score):
                        for frust_idx in frust_idx_set:
                            all_proposals.append(box.reshape(-1, 7))
                            all_proposal_scores.append(score.item())
                            all_proposal_cam_scores.append(cam_score)
                            all_proposals_idx.append(frust_idx)

            if len(all_proposals) == 0:
                continue

            # concatenate all frustum proposals
            all_proposals = torch.stack(all_proposals, dim=0)
            all_scores = torch.tensor(all_proposal_scores)
            all_cam_scores = torch.tensor(all_proposal_cam_scores)
            all_proposals_idx = torch.tensor(all_proposals_idx)

            # re-rank per frustum (>= mean score)
            for frust_idx in range(len(batch_labels)):
                frust_mask = (all_proposals_idx == frust_idx)
                frust_proposals = all_proposals[frust_mask]
                cur_frust_scores = all_scores[frust_mask]
                cur_cam_scores = all_cam_scores[frust_mask]
                frust_label = batch_labels[frust_idx]

                if len(cur_frust_scores) >= 1:
                    valid_score = cur_frust_scores.mean()

                    valid_mask = cur_frust_scores >= valid_score
                    valid_proposals = frust_proposals[valid_mask]

                    # use camera score
                    for prop, score in zip(valid_proposals, cur_cam_scores[valid_mask]):
                        proposal_boxes.append(prop)
                        frust_scores.append(score)
                        frust_batch_idx.append(b)
                        frust_labels.append(frust_label)
                else:
                    for prop, score in zip(frust_proposals, cur_cam_scores):
                        proposal_boxes.append(prop)
                        frust_scores.append(score)
                        frust_batch_idx.append(b)
                        frust_labels.append(frust_label)

        # DO STUFF
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

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

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
        batch_size = batch_dict['batch_size']

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