import copy
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
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
import cv2
from ..preprocessed_detector import PreprocessedDetector, PreprocessedGLIP

color_palette = [
    (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),  # Blue
    (1.0, 0.4980392156862745, 0.0),  # Orange
    (0.2, 0.6274509803921569, 0.17254901960784313),  # Green
    (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),  # Red
    (0.41568627450980394, 0.23921568627450981, 0.6039215686274509),  # Purple
    (0.6941176470588235, 0.23921568627450981, 0.16470588235294117),  # Brown
    (0.984313725490196, 0.6039215686274509, 0.6000000000000000),  # Pink
    (0.4, 0.4, 0.4),  # Gray
    (0.6509803921568628, 0.807843137254902, 0.8901960784313725),  # Teal
    (1.0, 1.0, 0.6000000000000000),  # Yellow
    (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),  # Light Green
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # Dark Purple
    (1.0, 0.7333333333333333, 0.47058823529411764),  # Light Orange
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098),  # Light Purple
    (0.5529411764705883, 0.8274509803921568, 0.7803921568627451)  # Turquoise
]

EDGES = [(0, 1), (4, 5), (0, 4), (1, 2), (5, 6), (1, 5), (2, 3), (6, 7), (2, 6), (3, 0), (7, 4), (3, 7), (0, 5), (1, 4)]

def draw_corners_on_cv(corners, image, color=(1, 1, 1), line_width=2, label='', max_num=500, tube_radius=None, x_offset=0, image_size=[900, 1600]):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    # coord_max = [x for x in reversed(image_size)]
    coord_max = image_size

    if isinstance(color[0], float):
        color = tuple(int(x*255) for x in color)

    def on_image(coord):
        return coord[0] >= 0 and coord[1] >= 0 and coord[0] <= coord_max[0] and coord[1] <= coord_max[1]

    def clip_line_to_image(start, end):
        start_valid = on_image(start)
        end_valid = on_image(end)

        # dont have to interpret
        if start_valid and end_valid:
            return start, end


        
        # end -> start = (start - end)
        vec = (end - start)

        # 0 < start[0] + vec[0] * p < im_size[1]
        # -start[0] < vec[0] * p < im_size[1] - start[0]
        # -start[0] / vec[0] < p < (im_size[1] - start[0]) / vec[0]
        dim_mins = []
        dim_maxs = []
        for d in range(2):
            p1 = (-start[d] / vec[d]).clamp(0, 1)
            p2 = ((coord_max[d] - start[d]) / vec[d]).clamp(0, 1)

            dim_mins.append(min(p1, p2))
            dim_maxs.append(max(p1, p2))

        vmin = max(dim_mins)
        vmax = min(dim_maxs)

        new_start = start + vmin * vec
        new_end = start + vmax * vec

        # check if valid (could be that this line does not intersect!)
        if not on_image(new_start) or not on_image(new_end):
            return None, None

        return new_start, new_end

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            image = cv2.line(image, (int(start[1] + x_offset), int(start[0])), (int(end[1] + x_offset), int(end[0])), color, line_width, lineType=cv2.LINE_AA) 

        i, j = k + 4, (k + 1) % 4 + 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            image = cv2.line(image, (int(start[1] + x_offset), int(start[0])), (int(end[1] + x_offset), int(end[0])), color, line_width, lineType=cv2.LINE_AA) 

        i, j = k, k + 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            image = cv2.line(image, (int(start[1] + x_offset), int(start[0])), (int(end[1] + x_offset), int(end[0])), color, line_width, lineType=cv2.LINE_AA) 


    i, j = 0, 5
    start, end = clip_line_to_image(corners[i], corners[j])
    if start is not None:
        image = cv2.line(image, (int(start[1] + x_offset), int(start[0])), (int(end[1] + x_offset), int(end[0])), color, line_width, lineType=cv2.LINE_AA) 

    i, j = 1, 4
    start, end = clip_line_to_image(corners[i], corners[j])
    if start is not None:
        image = cv2.line(image, (int(start[1] + x_offset), int(start[0])), (int(end[1] + x_offset), int(end[0])), color, line_width, lineType=cv2.LINE_AA) 

    return image

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
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
        lq=0.336, uq=0.356, iou_w=0.95, dst_w=0.226, dns_w=0.05, \
                 min_cam_iou=0.3, size_min=0.957, size_max=1.2, ry_min=0.0, ry_max=torch.pi, cq=0.46, num_mags=6,
                 max_dist=50, num_sizes=4, num_rotations=10, topk=1, nms_2d=0.7, nms_3d=1.0, score_thr=0.1, nms_normal=0.7, clamp_bottom=0
    ):
        super(FrustumProposerOG, self).__init__()

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

        self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

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

        self.box_fmt = model_cfg.get('BOX_FORMAT', 'xyxy')
        preds_path = model_cfg.get('PREDS_PATH', '/home/uqdetche/GLIP/jsons/OWL_')

        print('preds_path', preds_path)
        if 'PreprocessedGLIP' in preds_path:
            self.image_detector = PreprocessedGLIP(class_names=class_names)
        else:
            # exit()
            # preds_path = '/home/uqdetche/OpenPCDet/tools/coco_val2_'
            # preds_path = '/home/uqdetche/GLIP/OWL_'
            # preds_path = '/home/uqdetche/GLIP/jsons/OWL_'
            camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
            preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

            preds_paths = model_cfg.get('PREDS_PATHS', preds_paths)
            print('preds_paths', preds_paths)
            self.image_detector = PreprocessedDetector(preds_paths, class_names=class_names)

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

        print('pc bev pos range', pc_bev_pos_range)

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
    
    def calc_occl_scores(self, anchor: Tensor, box_proposals: Tensor, points: Tensor, dirs: Tensor, mags: Tensor):
        box_proposals = box_proposals.reshape(-1, 7)
        nboxes = box_proposals.shape[0]

        occl_scores = points.new_zeros((nboxes,))

        # phi = anchor.min() / 2.0
        # mags_empty = mags - phi
        # mags_occl = mags + phi

        # generate query points
        # empty_points = dirs * mags_empty
        # occl_points = dirs * mags_occl

        for i in range(nboxes):
            corners = boxes_to_corners_3d(box_proposals[i].reshape(1, 7)).reshape(8, 3)
            corners_mags = corners.norm(dim=-1, keepdim=True)
            corners_dirs = corners / corners_mags
            m1, m2 = corners_mags.min(), corners_mags.max()

            corners_mindir = corners_dirs.min(dim=0).values.reshape(1, 3)
            corners_maxdir = corners_dirs.max(dim=0).values.reshape(1, 3)


            # cur_mask = (dirs * corners_mindir).sum(dim=-1)
            # cur_mask = (dirs > corners_mindir).all(dim=1) & (dirs <= corners_maxdir).all(dim=1)
            # print('cur mask', cur_mask.shape, cur_mask.sum())
            
            # if cur_mask.sum() == 0:
                # continue

            cur_points = points#[cur_mask]
            # cur_dirs = dirs#[cur_mask]
            cur_mags = mags#[cur_mask]
            

            real_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.reshape(1, -1, 3), box_proposals[[i]].reshape(1, -1, 7))
            real_mask = (real_point_box_indices >= 0).reshape(-1)
            num_real = real_mask.sum()

            # find points which might occur before the box -> not in box, and magnitude before
            # probably_before_box = (cur_mags <= m2) & (~real_mask)

            # print('before', probably_before_box.sum())

            # very rough
            # num_occl = (cur_mags > m2).sum()
            # num_empty = (cur_mags < m1).sum()
            # num_empty = probably_before_box.sum()
            # real points score

            num_fail = ((cur_mags > m1) & (~real_mask)).sum()

            # empty points
            # empty_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(empty_points.reshape(1, -1, 3), box_proposals[i].reshape(1, -1, 7))

            # # occl points
            # occl_point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(occl_points.reshape(1, -1, 3), box_proposals[i].reshape(1, -1, 7))

            # empty_mask = (empty_point_box_indices >= 0).reshape(-1)
            # num_empty = empty_mask.sum()

            # occl_mask = (occl_point_box_indices >= 0).reshape(-1)
            # num_occl = occl_mask.sum()
            
            occl_scores[i] = num_fail
            # occl_scores[i] = (num_fail - 2*num_real)# / (points.shape[0]*2)
            # occl_scores[i] = num_occl + num_empty
        
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
        img_aug_matrix = None if 'img_aug_matrix' not in batch_dict else batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        # lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = None if img_aug_matrix is None else img_aug_matrix[..., :3, :3]
        post_trans = None if img_aug_matrix is None else img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # 2d multiview detections loaded (loaded from coco jsons)
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        proposal_boxes = []
        proposal_scores = []
        frust_labels = []
        frust_batch_idx = []
        frust_scores = []
        frust_indices = []

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]
            # plane_mask = fit_plane(cur_points)
            # # print('num in plane', plane_mask.shape, plane_mask.sum(), plane_mask.sum() / cur_points.shape[0])
            # non_ground = torch.bitwise_not(plane_mask)
            # cur_points = cur_points[non_ground]
            foreground_pts = cur_points

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            batch_frusts = []
            batch_raw_frusts = []
            batch_cam_boxes = []
            batch_frusts_cams = []
            batch_frusts_points = []
            batch_weighted_centres = []
            batch_bev_masks = []
            batch_scores = []
            batch_labels = []

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                if cam_boxes.shape[0] > 0:
                    selected = batched_nms(cam_boxes, cam_scores, cam_labels, self.nms_2d)
                    cam_boxes, cam_labels, cam_scores = cam_boxes[selected], cam_labels[selected], cam_scores[selected]

                cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                cam_points = cam_points[cam_mask]

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < self.score_thr:
                        continue

                    if self.box_fmt == 'xyxy':
                        x1, y1, x2, y2 = box.cpu()
                    else:
                        box[..., 2:] += box[..., 0:2]
                        x1, y1, x2, y2 = box

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

                        weighted_centre_xyz = self.get_geometry_at_image_coords(weighted_centre_cam.reshape(-1, 3), [c], [b], 
                            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                        )
                        # weighted_centre_cam = torch.quantile(box_points, q=0.5, dim=0)
                    else:
                        # print('no pts in box')
                        
                        continue
                        cur_frustum_min = self.frustum_min
                        cur_frustum_max = self.frustum_max
                        weighted_centre_xyz = None



                    cur_frustum_max = torch.minimum(cur_frustum_max, self.frustum_max)
                    cur_frustum_min = torch.maximum(cur_frustum_min, self.frustum_min)

                    # if self.clamp_bottom: # TODO
                    #     old_mask_num = on_box.sum().item()
                    #     on_box = (on_box & (cam_points[..., 2] >= cur_frustum_min) & (cam_points[..., 2] <= cur_frustum_max))
                    #     print('old mask num', old_mask_num, 'new', on_box.sum().item())

                    box = box.cuda()
                    xyzxyz = torch.cat([box[0][None], box[1][None], cur_frustum_min[None], box[2][None], box[3][None], cur_frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    if self.nms_3d > 0:
                        bev_mask = self.get_frustum_bev_mask(frust_box)

                    if self.SAVE_BLEND_FILES:
                        d1 = box_points[:, 2].min()
                        # d2 = box_points[:, 2].max()
                        d2 = torch.quantile(box_points[:, 2], 0.9)
                        xyzxyz = torch.cat([box[0][None], box[1][None], torch.ones_like(cur_frustum_max[None])*d1, box[2][None], box[3][None], torch.ones_like(cur_frustum_max[None])*d2])
                        # z1 = box_points[:, 2].min()
                        # z2 = z1 + torch.minimum()
                        # xyzxyz = torch.cat([box[0][None], box[1][None], cur_frustum_min[None], box[2][None], box[3][None], cur_frustum_max[None]])

                        frust_box_raw = get_cam_frustum(xyzxyz)
                        frust_box_raw = self.get_geometry_at_image_coords(frust_box_raw, [c] * 8, [b] * 8, # 8 corners in a box
                            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                        )

                        batch_raw_frusts.append(frust_box_raw)

                    batch_frusts.append(frust_box)
                    batch_cam_boxes.append(box)
                    batch_frusts_cams.append(c)
                    batch_frusts_points.append(box_points)
                    batch_weighted_centres.append(weighted_centre_xyz)
                    if self.nms_3d > 0:
                        batch_bev_masks.append(bev_mask[None])
                    batch_scores.append(score[None])
                    batch_labels.append(label[None])

            if len(batch_frusts) == 0:
                proposal_boxes = torch.zeros((0, 7), device='cuda')
                frust_labels = torch.zeros((0), dtype=torch.long)
                frust_scores = torch.zeros((0))
                frust_batch_idx = torch.zeros((0), dtype=torch.long)

                return proposal_boxes, frust_labels, frust_scores, frust_batch_idx
            
            
            ############################################# BLENDER VISUALISATION ####################################### 
            if self.SAVE_BLEND_FILES:
                top_proposals = []
                top_proposals_scores = []
                images = batch_dict['camera_imgs']
                height, width = images.shape[-2:]
                frust_boxes = torch.stack(batch_frusts, dim=0).clone().cpu().numpy()
                batch_raw_frusts = torch.stack(batch_raw_frusts, dim=0)
                # grid = cam_points[..., :2].clone().reshape(1, 1, -1, 2)
                pt_colors = torch.ones_like(cur_points)
                print('image range', images.min(), images.max())
                # for c in range(6):
                #     cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                #     print('cam_pts', cam_points.shape, cam_mask.shape)
                #     cam_points = cam_points[cam_mask]
                #     for i in range(2):
                #         print('cam pts', i, cam_points[..., i].min(), cam_points[..., i].max())
                #     grid = cam_points[..., :2].clone()
                #     # swap dims
                #     grid[..., 1] = cam_points[..., 0].clone()
                #     grid[..., 0] = cam_points[..., 1].clone()
                #     grid = grid.reshape(1, 1, -1, 2)
                #     grid[..., 0]  = (grid[..., 0] / height) * 2.0 - 1.0
                #     grid[..., 1] = (grid[..., 1] / width) * 2.0 - 1.0
                #     # grid[..., ::-1] = grid
                            
                #     sampled_image = F.grid_sample(images[:, c], grid=grid)
                #     sampled_image = sampled_image.reshape(3, -1).permute(1, 0)

                #     # set colors for those seen
                #     print('pt_colors', pt_colors.shape, cam_mask.shape, sampled_image.shape)
                #     pt_colors[cam_mask.reshape(-1)] = sampled_image
                

                if 'gt_boxes' in batch_dict:
                    gt_boxes = batch_dict['gt_boxes'][b]
                    gt_labels = gt_boxes[..., -1].long()



                    for i in range(gt_labels.shape[0]):
                        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.unsqueeze(0), gt_boxes[[i], :7].reshape(1, -1, 7))
                        point_box_indices = point_box_indices.reshape(-1)
                        pt_mask = (point_box_indices == i)
                        color = torch.tensor([x for x in color_palette[gt_labels[i].item() % len(color_palette)]], device=pt_colors.device)
                        pt_colors[pt_mask] = color.reshape(1, 3)
                        # num_pts_in_boxes[i] = (point_box_indices == i).sum()


                    print('gt_boxes', gt_boxes.shape)
                    gt_boxes_np = gt_boxes.cpu().numpy()

                    np.save('nusc_vis_gt_boxes', gt_boxes_np)

                print('pt_colors',pt_colors.min(), pt_colors.max())

                np.save('nusc_vis_col', pt_colors.cpu().numpy())
                np.save('nusc_vis_pcl', cur_points.cpu().numpy())
                np.save('nusc_vis_frust_boxes', frust_boxes)
                np.save('nusc_vis_raw_frust_boxes', batch_raw_frusts.cpu().numpy())


            ############################################# BLENDER VISUALISATION ####################################### 


            # do frustum NMS to combine frustums that align well (as objects can appear on multiple cameras)
            batch_scores = torch.cat(batch_scores, dim=0)

            N = batch_scores.shape[0]
            indices = torch.arange(0, N)

            # if self.nms_3d > 0:
            # batch_bev_masks = torch.cat(batch_bev_masks, dim=0)

            # # frustum nms
            # indices, combined = self.frustum_bev_nms(batch_bev_masks, batch_scores, batch_labels, nms_thresh=self.nms_3d)

            # # combine kept frustums with the ones that were removed (e.g. if an object is over two views, need to combine the frustums)
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

            for frust_idx, (frust_box, c, box, box_points, weighted_centre_xyz, label, score) in \
                    enumerate(zip(batch_frusts, batch_frusts_cams, batch_cam_boxes, \
                                                        batch_frusts_points, batch_weighted_centres, batch_labels, batch_scores)):
                if frust_idx not in indices:
                    continue

                # project image points back to lidar
                box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                    camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                    post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                )

                if self.clamp_bottom > 0:
                    for d in range(3):
                        f1, f2 = box_points_xyz[..., d].min(), box_points_xyz[..., d].max()
                        f1_, f2_ = frust_box[..., d].min(), frust_box[..., d].max(),

                        f1 = torch.maximum(f1, f1_)
                        f2 = torch.minimum(f2, f2_)
                        # print('old range, new', (z1_, z2_), (z1, z2))

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

                if self.search_depth is not None:
                    center_vec = (center_vec / center_vec.norm()) * self.search_depth #*self.anchors[label.item() - 1, :2].norm()

                if not self.rand_center:
                    bev_pts_xyz = frust_center_close.reshape(1, 3) + center_vec * mags
                else:
                    bev_pts_xyz = weighted_centre_xyz.reshape(1, 3) + torch.randn((self.num_mags, 3), dtype=torch.float32, device=weighted_centre_xyz.device)

                # print('bev pts', bev_pts_xyz)

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
                if not self.MULTICAM_IOU:
                    ious = self.calc_iou(batch_dict, c, b, box, curr_corner_proposals)
                else:
                    multicam_idx = [i for i, x in enumerate(batch_labels) if x == label]
                    ious = self.multicam_ious(batch_dict, [batch_frusts_cams[i] for i  in multicam_idx], b, [batch_cam_boxes[i] for i in multicam_idx], curr_corner_proposals)

                if weighted_centre_xyz is not None:
                    dists = torch.cdist(weighted_front_centres.reshape(-1, 3), weighted_centre_xyz.reshape(1, 3)).reshape(-1)
                    # print('dists', dists)
                    # print('dists', dists.min(), dists.max())
                    dists_ranked = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
                    # dists_ranked = dists / dists.sum()
                    dists_ranked = 1 - dists_ranked


                    # print('dists_ranked', dists_ranked)
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
                    continue

                # # project image points back to lidar
                # box_points_xyz = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                #     camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                #     post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                # )
                
                # calculate how many in each proposal
                num_pts_in_boxes = torch.zeros(curr_box_proposals.shape[0])
                aln_vals = torch.zeros(curr_box_proposals.shape[0])

                if box_points_xyz.shape[0] > 0:
                    # point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(box_points_xyz.unsqueeze(0), curr_box_proposals.reshape(1, -1, 7))
                    # print('point_box_indices entire', point_box_indices.shape)
                    # point_box_indices = point_box_indices.reshape(-1)
                    # print('box indices', point_box_indices.min(), point_box_indices.max())
                    # print('point_box_indices', point_box_indices.shape, curr_box_proposals.shape, box_points_xyz.shape)

                    for i in range(curr_box_proposals.shape[0]):
                        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(box_points_xyz.unsqueeze(0), curr_box_proposals[[i]].reshape(1, -1, 7))
                        num_pts_in_boxes[i] = (point_box_indices >= 0).sum()


                        if self.SAVE_BLEND_FILES:
                            in_box, box_rel_pts = self.points_in_boxes(box_points_xyz, curr_box_proposals[[i]])

                            # check if there is an issue
                            if abs(in_box.sum() - num_pts_in_boxes[i]) > 5 and num_pts_in_boxes[i] > 0:
                                box_rel_pts = box_rel_pts.reshape(-1, 3)

                                box_rel = box_rel_pts[in_box.reshape(-1)]

                                print('in pts', box_points_xyz[in_box.reshape(-1)], 'box', curr_box_proposals[i])
                                print('in_box', in_box.sum(), 'num_pts_in_boxes[i]', num_pts_in_boxes[i])
                                in_pts = box_points_xyz[in_box.reshape(-1)]
                                in_pts2 = box_points_xyz[point_box_indices.reshape(-1) == i]
                                print('in_pts', in_pts.shape, 'in_pts2', in_pts2.shape)
                                corners = curr_corner_proposals[i]
                                
                                from matplotlib import pyplot as plt


                                fig = plt.figure(figsize=(15,5))
                                ax1 = fig.add_subplot(1, 3, 1, projection='3d')
                                ax2 = fig.add_subplot(1, 3, 2, projection='3d')
                                ax3 = fig.add_subplot(1, 3, 3, projection='3d')
                                for (ei, ej) in EDGES:
                                    ax1.plot([corners[ei, 0].item(), corners[ej, 0].item()], [corners[ei, 1].item(), corners[ej, 1].item()], [corners[ei, 2].item(), corners[ej, 2].item()], label='corners', color='red')
                                    ax2.plot([corners[ei, 0].item(), corners[ej, 0].item()], [corners[ei, 1].item(), corners[ej, 1].item()], [corners[ei, 2].item(), corners[ej, 2].item()], label='corners', color='red')
                                ax1.scatter(in_pts[..., 0].cpu().numpy(), in_pts[..., 1].cpu().numpy(), in_pts[..., 2].cpu().numpy(), label='my code')
                                ax2.scatter(in_pts2[..., 0].cpu().numpy(), in_pts2[..., 1].cpu().numpy(), in_pts2[..., 2].cpu().numpy(), label='their code')
                                ax3.scatter(box_rel[..., 0].cpu().numpy(), box_rel[..., 1].cpu().numpy(), box_rel[..., 2].cpu().numpy(), label='my code (box rel)')

                                r1, r2 = in_pts.min().item(), in_pts.max().item()
                                for ax in [ax1, ax2]:
                                    ax.set_xlim(r1, r2)
                                    ax.set_ylim(r1, r2)
                                    ax.set_zlim(r1, r2)

                                r1, r2 = box_rel.min().item(), box_rel.max().item()
                                ax3.set_xlim(r1, r2)    
                                ax3.set_ylim(r1, r2)    
                                ax3.set_zlim(r1, r2)

                                ax1.set_title('my code')
                                ax2.set_title('their code')
                                ax3.set_title('relative')

                                plt.savefig('inpts.png')

                                exit()


                        if self.aln_w > 0:
                            if num_pts_in_boxes[i] > 3:
                                in_points = box_points_xyz[point_box_indices == i]

                                _, _, v = torch.pca_lowrank(in_points, q=None, center=True, niter=2)

                                v = v[:, 1] / v[:, 1].norm() 
                                aln_vals[i] = v[0] * torch.cos(curr_box_proposals[i, -1]) + v[1] * torch.sin(curr_box_proposals[i, -1])

                soft_densities = (num_pts_in_boxes) / (num_pts_in_boxes.max() + 1e-8)
                # soft_densities = num_pts_in_boxes / (num_pts_in_boxes.sum() + 1e-8)
                # second_stage_scores = ((1.0 - self.dns_w) + soft_densities * self.dns_w) * ((1.0 - self.iou_w) + ious * self.iou_w) * ((1.0 - self.dst_w) + dists_ranked.cpu() * self.dst_w)
                
                if not self.MULT:
                    second_stage_scores = soft_densities * self.dns_w  + ious * self.iou_w + dists_ranked.cpu() * self.dst_w
                else:
                    second_stage_scores = soft_densities * self.dns_w * ious * self.iou_w * dists_ranked.cpu() * self.dst_w


                if self.aln_w > 0:
                    second_stage_scores += aln_vals * self.aln_w

                if self.occl_w > 0:
                    pts_mags = box_points_xyz.norm(dim=-1, keepdim=True)
                    pts_dirs = box_points_xyz / pts_mags
                    occl_scores = self.calc_occl_scores(self.anchors[label.item() - 1], curr_box_proposals, box_points_xyz, pts_dirs, pts_mags)
                    occl_ranked = occl_scores / (occl_scores.max() + 1e-6)
                    occl_ranked = 1 - occl_ranked

                    second_stage_scores += self.occl_w * occl_ranked.cpu()

                if self.ego_w > 0:
                    ego_dists = curr_box_proposals[..., :3].norm(dim=-1)
                    ego_ranks = ego_dists / ego_dists.max()

                    second_stage_scores += self.ego_w * ego_ranks.to(second_stage_scores.device)

                if self.OCCL_MULT:
                    pts_mags = box_points_xyz.norm(dim=-1, keepdim=True)
                    pts_dirs = box_points_xyz / pts_mags
                    occl_scores = self.calc_occl_scores(self.anchors[label.item() - 1], curr_box_proposals, box_points_xyz, pts_dirs, pts_mags)
                    # print('densities', soft_densities.shape, ious.shape, occl_scores.shape)
                    second_stage_scores = soft_densities * ious * occl_scores.to(ious.device)

                # NMS and ordering for topk
                selected, _ = iou3d_nms_utils.nms_normal_gpu(curr_box_proposals, second_stage_scores, thresh=self.nms_normal)
                curr_box_proposals = curr_box_proposals[selected].contiguous()
                curr_corner_proposals = curr_corner_proposals[selected]
                ious = ious[selected]
                second_stage_scores = second_stage_scores[selected]

                if self.SAVE_BLEND_FILES:
                    top_proposals.append(curr_box_proposals)
                    top_proposals_scores.append(second_stage_scores)

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
                    frust_indices.append(frust_idx)
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

        ############################################# BLENDER VISUALISATION ####################################### 
        if self.SAVE_BLEND_FILES:
            folder = '/home/uqdetche/OpenPCDet/tools/paper_scripts/frust_vis/'

            # wanted_point = proposal_boxes.new_tensor([-2, 17, 0.0]) # trailer not very visible

            # wanted_point = proposal_boxes.new_tensor([-10, 10, 0.0]) # trailer not very visible
            wanted_point = proposal_boxes.new_tensor([-5, -10, 0.0]) # truck not very visible
            # wanted_point = proposal_boxes.new_tensor([-10, 0, 0.0])
            dists = torch.cdist(proposal_boxes[:, :3], wanted_point.unsqueeze(0))

            dists = dists.reshape(-1)
            min_idx = torch.argmin(dists)

            print('dists', dists.shape, len(proposal_boxes), len(batch_frusts), len(batch_raw_frusts), len(top_proposals))
            print('frust_indices', len(frust_indices), frust_indices)
            frust_idx = frust_indices[min_idx.item()]
            
            print('proposal_boxes', proposal_boxes.shape)
            proposal = proposal_boxes[min_idx].cpu().numpy()
            frust = batch_frusts[frust_idx]
            top5 = top_proposals[min_idx.item()][:5]
            top5_scores = top_proposals_scores[min_idx.item()][:5]
            raw_frust = batch_raw_frusts[frust_idx]
            cam_box = batch_cam_boxes[frust_idx]

            wc = batch_weighted_centres[frust_idx]
            print('wc', wc)

            pts = batch_frusts_points[frust_idx]
            c = batch_frusts_cams[frust_idx]
            print('label', batch_labels[frust_idx])
            lq = torch.quantile(box_points[:, 2], self.lq)
            uq = torch.quantile(box_points[:, 2], self.uq)
            cq = torch.quantile(box_points[:, 2], self.cq)

            # project image points back to lidar
            pts = self.get_geometry_at_image_coords(pts, [c] * pts.shape[0], [0] * pts.shape[0],
                camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
            )

            # rotate to make in frustum coords
            frust_bev_box = torch.cat([frust[[2*i, 2*i+1], :].mean(dim=0)[None, :] for i in range(4)], dim=0)

            frust_center_close = frust_bev_box[:2, :].mean(dim=0)
            frust_center_far = frust_bev_box[2:, :].mean(dim=0)

            frust_vec = (frust_center_far - frust_center_close)
            frust_rot = torch.atan2(frust_vec[1], frust_vec[0])

            # pts = common_utils.rotate_points_along_z(pts[None], frust_rot[None])[0]
            # frust = common_utils.rotate_points_along_z(frust[None], frust_rot[None])[0]
            # raw_frust = common_utils.rotate_points_along_z(raw_frust[None], frust_rot[None])[0]
            
            # # rotate proposals
            # top5[:, 0:3] = common_utils.rotate_points_along_z(top5[None, :, 0:3], frust_rot[None])[0]
            # top5[:, 6] += frust_rot

            print('pts rot', pts.shape)
            print('frust rot', frust.shape)
            print('top5 props', top5.shape)

            # density plot?
            x1, x2 = raw_frust[..., 0].min().item(), raw_frust[..., 0].max().item()
            y1, y2 = raw_frust[..., 1].min().item(), raw_frust[..., 1].max().item()
            z1, z2 = raw_frust[..., 2].min().item(), raw_frust[..., 2].max().item()

            xc, yc = (x1 + x2) / 2, (y1 + y2)/2
            l, w, h = (x2 - x1), (y2 - y1), (z2 - z1)
            x1, x2 = xc - l*0.6, xc + l *0.6
            y1, y2 = yc - w*0.6, yc + w*0.6

            # print('x1 x2', x1, x2)
            # x1, x2 = x2, x1

            plot_n = 20
            
            # np.histogram
            xs = np.linspace(x1, x2, plot_n)
            ys = np.linspace(y1, y2, plot_n)

            z = frust[..., 2].mean().item()

            dense_plot = np.zeros((plot_n, plot_n), dtype=np.float32)
            dist_plot = np.zeros((plot_n, plot_n), dtype=np.float32)
            iou_plot = np.zeros((plot_n, plot_n), dtype=np.float32)
            inlier_plot = np.zeros((plot_n, plot_n), dtype=np.float32)
            occl_plot = np.zeros((plot_n, plot_n), dtype=np.float32)


            multicam_idx = [i for i, x in enumerate(batch_labels) if x == batch_labels[frust_idx]]

            curr_anchor = self.anchors[batch_labels[frust_idx].item() - 1]
            cur_points_mags = torch.norm(cur_points, dim=-1, keepdim=True)
            cur_points_dirs = cur_points / cur_points_mags

            # cur_points, cur_points_dirs, cur_points_mags
            pts_mags = pts.norm(dim=-1, keepdim=True)
            pts_dirs = pts / pts_mags
                
            for i in range(plot_n):
                for j in range(plot_n):
                    x, y = xs[i], ys[j]

                    # dist_plot[j, i] = np.sqrt(x**2 + y**2)
                    # dist_plot[j, i] = np.sqrt((x-wc[..., 0].item())**2 + (y - wc[..., 1].item())**2)
                    # dist_plot[j, i] = ((x-wc[..., 0].item())**2 + (y - wc[..., 1].item())**2)**2
                    dist_plot[j, i] = (x**2 + y**2)


                    # box = torch.tensor([x, y, z, l, w, h, 0.0], dtype=pts.dtype, device=pts.device)
                    box = self.base_boxes[batch_labels[frust_idx].item() - 1].clone()
                    box[..., 0] += x
                    box[..., 1] += y
                    box[..., 2] += z
                    # corners = boxes_to_corners_3d(box.reshape(1, -1))
                    # print('corner')
                    max_pts = 0
                    for idx in range(box.shape[0]):
                        # point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(pts.unsqueeze(0), box[[idx]].reshape(1, -1, 7))
                        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.unsqueeze(0), box[[idx]].reshape(1, -1, 7))
                        num_pts = (point_box_indices >= 0).sum().item()

                        if num_pts > max_pts:
                            max_pts = num_pts

                    dense_plot[j, i] = max_pts

                    corners = boxes_to_corners_3d(box)
                    # ious = self.multicam_ious(batch_dict, [batch_frusts_cams[i] for i  in multicam_idx], b, [batch_cam_boxes[i] for i in multicam_idx], corners)
                    # iou_plot[j, i] = ious.max().item()

                    iou_plot[j, i] = self.calc_iou(batch_dict, c, b, cam_box, corners).max().item()
                    # if x > x1 and x < x2 and y > y1 and y < y2:
                    # occl_plot[j, i] = self.calc_occl_scores(curr_anchor, box, cur_points, cur_points_dirs, cur_points_mags).min().item()
                    occl_plot[j, i] = self.calc_occl_scores(curr_anchor, box, pts, pts_dirs, pts_mags).min().item()
                    # inlier_plot[j, i] = self.calc_inlier_scores(curr_anchor, box, cur_points, cur_points_dirs, cur_points_mags).max().item()


            dense_plot = (dense_plot - dense_plot.min()) / (dense_plot.max() - dense_plot.min() + 1e-6)

            # dist_plot = (dist_plot - dist_plot.min()) / (dist_plot.max() - dist_plot.min() + 1e-6)
            dist_plot = dist_plot / dist_plot.max()
            dist_plot = 1 - dist_plot

            # occl_plot = occl_plot / occl_plot.max()
            occl_plot = (occl_plot - occl_plot.min()) / (occl_plot.max() - occl_plot.min() + 1e-6)
            occl_plot = 1 - occl_plot 
            # inlier_plot = (inlier_plot - inlier_plot.min()) / (inlier_plot.max() - inlier_plot.min() + 1e-6)


            # final_plot = self.iou_w * iou_plot + self.dns_w * dense_plot + self.dst_w * dist_plot
            final_plot = iou_plot + self.dns_w * dense_plot #+ self.dst_w * dist_plot
            # final_plot = (final_plot - final_plo)
            # final_plot = iou_plot * dense_plot * occl_plot

            # dense_plot = np.sqrt(dense_plot)




            from matplotlib import pyplot as plt
            # fig = plt.figure(figsize=(15,5))
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax3 = fig.add_subplot(1, 3, 3)
            # fig, (ax2, ax1, ax3, ax4, ax5) = plt.subplots(ncols=5, sharey=True, sharex=True, figsize=(15, 4), frameon=False)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(9, 4), frameon=False)

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            # ax4.axis('off')
            # ax5.axis('off')
            # ax1.set_xticklabels([f'{x:.2f}' for x in xs])
            # ax1.set_yticklabels([f'{y:.2f}' for y in ys])

            cmap = 'copper'
            ax1.imshow(dense_plot, cmap=cmap)
            ax1.set_title('Density $\mathfrak{C}_1$')
            # ax2.imshow(dist_plot, cmap=cmap)
            # ax2.set_title('Ego Distance $\mathfrak{C}_1$')

            ax2.imshow(iou_plot, cmap=cmap)
            ax2.set_title('IoU $\mathfrak{C}_2$')

            # ax4.imshow(occl_plot, cmap=cmap)
            # ax4.set_title('occl')

            ax3.imshow(final_plot, cmap=cmap)
            ax3.set_title('$\mathfrak{C} = \mathfrak{C}_1 + \mathfrak{C}_2$')

            def corners_to_idx(corners):
                corners_idx = []
                for i in range(8):
                    x, y = corners[i, 0].item(), corners[i, 1].item()
                    
                    if x < x1 or x > x2 or y < y1 or y > y2:
                        return None

                    xidx = ((x - x1) / (x2 - x1))*plot_n
                    yidx = ((y - y1) / (y2 - y1))*plot_n
                    # xidx = np.argmin(np.abs(xs - x))
                    # yidx = np.argmin(np.abs(ys - y))
                    corners_idx.append((xidx, yidx))

                return corners_idx

            gt_boxes = batch_dict['gt_boxes'][b]
            gt_dists = torch.cdist(wanted_point.unsqueeze(0), gt_boxes[..., :3]).reshape(-1)
            # print('gt_dists', gt_dists.shape)
            gt_boxes = gt_boxes[torch.argmin(gt_dists).reshape(-1)].reshape(1, 10)
            # print('gt_boxes', gt_boxes.shape)
            gt_corners_tens = boxes_to_corners_3d(gt_boxes)


            frust_dense = corners_to_idx(raw_frust)
            gt_corners = [corners_to_idx(corners) for corners in gt_corners_tens]
            gt_corners = filter(lambda x: x is not None, gt_corners)

            top5_corners_tens = boxes_to_corners_3d(top5)
            top5_corners = [corners_to_idx(x) for x in top5_corners_tens]
            top5_corners = filter(lambda x: x is not None, top5_corners)

            lightred = (255/255, 127/255, 127/255)
            lightred = 'red'


            for (ei, ej) in [(1, 2), (5, 6), (1, 5), (2, 6)]:

            #     print(f'{frust_dense[ei]} -> {frust_dense[ej]}')
                ax1.plot([frust_dense[ei][0], frust_dense[ej][0]], [frust_dense[ei][1], frust_dense[ej][1]], label='corners', color=lightred, lw=4, ls='--')
                ax2.plot([frust_dense[ei][0], frust_dense[ej][0]], [frust_dense[ei][1], frust_dense[ej][1]], label='corners', color=lightred, lw=4, ls='--')
                ax3.plot([frust_dense[ei][0], frust_dense[ej][0]], [frust_dense[ei][1], frust_dense[ej][1]], label='corners', color=lightred, lw=4, ls='--')
            #     ax2.plot([raw_frust[ei][0].item(), raw_frust[ej][0].item()], [raw_frust[ei][1].item(), raw_frust[ej][1].item()], label='corners', color='red')
            
            for rnk, (gt, score) in enumerate(zip(top5_corners, top5_scores)):
                for (ei, ej) in EDGES:
                    ax1.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='green', lw=2)
                    ax2.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='green', lw=2)
                    ax3.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='green', lw=2)

                # if rnk == 0:
                #     ax1.text(max(x[0] for x in gt), max(x[1] for x in gt), s='$\mathfrak{C}=' + f'{score:.2f}$')
                #     ax2.text(max(x[0] for x in gt), max(x[1] for x in gt), s='$\mathfrak{C}=' + f'{score:.2f}$')
                #     ax3.text(max(x[0] for x in gt), max(x[1] for x in gt), s='$\mathfrak{C}=' + f'{score:.2f}$')

            for gt in gt_corners:
                for (ei, ej) in EDGES:
                    ax1.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='blue', lw=2)
                    ax2.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='blue', lw=2)
                    ax3.plot([gt[ei][0], gt[ej][0]], [gt[ei][1], gt[ej][1]], label='corners', color='blue', lw=2)

            ax1.invert_xaxis()
            ax2.invert_xaxis()
            ax3.invert_xaxis()

            plt.savefig(f'{folder}/criteria.png', bbox_inches='tight', dpi=1000)

            # save
            np.save(f'{folder}/top5_proposals', top5.cpu().numpy())
            # np.save(f'{folder}/pts', pts.cpu().numpy())
            np.save(f'{folder}/pts', cur_points.cpu().numpy())
            np.save(f'{folder}/frust', frust.cpu().numpy())
            np.save(f'{folder}/raw_frust', raw_frust.cpu().numpy())
            # np.save(f'{folder}/lq', frust.cpu().numpy())
            # np.save(f'{folder}/proposals', proposal_boxes.cpu().numpy())

            images_joined_np = batch_dict['camera_imgs'][b][c].cpu().permute(1, 2, 0).numpy()
            images_joined_cv = images_joined_np.copy()
            images_joined_cv = (images_joined_cv - images_joined_cv.min()) / (images_joined_cv.max() - images_joined_cv.min())

            images_joined_cv = (images_joined_cv *255).astype(np.uint8)
            images_joined_3d = images_joined_cv.copy()

            image_pos, _ = self.project_to_camera(batch_dict, top5_corners_tens.reshape(-1, 3), b, c)
            image_pos = image_pos[..., :2].reshape(-1, 8, 2)

            # clamp to image dimensions
            image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
            image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

            for idx in range(len(top5)):
                print('image_pos[idx]', image_pos[idx].shape)
                xy1 = image_pos[idx].min(dim=0).values
                xy2 = image_pos[idx].max(dim=0).values
                print('max', xy1.shape)
                print('min/max', xy1, xy2)
            

                images_joined_cv = cv2.rectangle(images_joined_cv, (int(xy1[0]), int(xy1[1])), (int(xy2[0]), int(xy2[1])), (0,255,0), 3) 
                images_joined_3d = draw_corners_on_cv(image_pos[idx, ..., [1, 0]], images_joined_3d, color=(0,255,0))

            image_pos, _ = self.project_to_camera(batch_dict, gt_corners_tens.reshape(-1, 3), b, c)
            image_pos = image_pos[..., :2].reshape(-1, 8, 2)

            # clamp to image dimensions
            image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
            image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

            for idx in range(len(gt_corners_tens)):
                xy1 = image_pos[idx].min(dim=0).values
                xy2 = image_pos[idx].max(dim=0).values

                color =(156,81,169)

                images_joined_cv = cv2.rectangle(images_joined_cv, (int(xy1[0]), int(xy1[1])), (int(xy2[0]), int(xy2[1])), color, 3) 
                images_joined_3d = draw_corners_on_cv(image_pos[idx, ..., [1, 0]], images_joined_3d, color=color)

            images_joined_cv = cv2.rectangle(images_joined_cv, (int(cam_box[0]), int(cam_box[1])), (int(cam_box[2]), int(cam_box[3])), (255, 255, 0), 3) 
            images_joined_3d = cv2.rectangle(images_joined_3d, (int(cam_box[0]), int(cam_box[1])), (int(cam_box[2]), int(cam_box[3])), (255, 255, 0), 3) 

            cv2.imwrite(f'{folder}/props_image.png', images_joined_cv[:, :, [2, 1, 0]])
            cv2.imwrite(f'{folder}/props_image_3d.png', images_joined_3d[:, :, [2, 1, 0]])

            # exit()

            # np.save('nusc_vis_final_boxes', proposal_boxes.cpu().numpy())
        ############################################# BLENDER VISUALISATION ####################################### 

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

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

    def multicam_ious(self, batch_dict, cs, b, boxes, curr_corner_proposals):
        # takes multicamera views into account
        ious_list = []
        for (c, box) in zip(cs, boxes):
            ious = self.calc_iou(batch_dict, c, b, box, curr_corner_proposals)
            ious_list.append(ious.unsqueeze(0))

        ious_per_box = torch.cat(ious_list, dim=0)
        # print('ious_per_box', ious_per_box.shape, ious_per_box.max(dim=0).values)
        # print('ious_per_box', ious_per_box)
        # ious = ious_per_box.max(dim=0).values
        # ious = ious_per_box.mean(dim=0)

        nonzero_sum = (ious_per_box > 0).sum(dim=0)
        ious = ious_per_box.sum(dim=0) / (nonzero_sum + 1e-6)

        return ious

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()

        img_aug_matrix = None if 'img_aug_matrix' not in batch_dict else batch_dict['img_aug_matrix']
        # img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        cur_img_aug_matrix = None if img_aug_matrix is None else img_aug_matrix[batch_idx, [cam_idx]]#.cpu().detach().clone()
        cur_lidar_aug_matrix = lidar_aug_matrix[batch_idx]#.cpu().detach().clone()
        cur_lidar2image = lidar2image[batch_idx, [cam_idx]]#.cpu().detach().clone()


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
        if img_aug_matrix is not None:
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

    def points_in_boxes(self, points, boxes3d) -> tuple:
        N = boxes3d.shape[0]

        points = points.clone()
        # instead of rotating the box, rotate the pointcloud (so can simply threshold the points)
        points = points[:, None, :].repeat(1, N, 1)
        points[..., :3] = points[..., :3] - boxes3d[None, :, 0:3] # centre the points
        points = points.permute((1, 0, 2))

        points = common_utils.rotate_points_along_z(points, - boxes3d[:, 6]) # rotate by negative angle

        l, w, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
        # x1, x2 = -l/2, l/2
        # y1, y2 = -w/2, w/2
        # z1, z2 = -h/2, h/2
        # print('x1, x2', x1, x2, y1, x2, z1, z2)

        in_box = (
            # (points[..., 0] >= x1) &
            (points[..., 0].abs() <= l/2) &
            
            # (points[..., 1] >= y1) &
            (points[..., 1].abs() <= w/2) &
            
            # (points[..., 2] >= z1) &
            (points[..., 2].abs() <= h/2)
            )

        # assert points.shape[-1] == 5

        return in_box, points

    def get_geometry_at_image_coords(self, image_coords, cam_idx, batch_idx, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        # image_coords like (L, 3) # 

        camera2lidar_rots = camera2lidar_rots[batch_idx, cam_idx].to(torch.float)
        camera2lidar_trans = camera2lidar_trans[batch_idx, cam_idx].to(torch.float)
        intrins = intrins[batch_idx, cam_idx].to(torch.float)

        if post_rots is not None:
            post_rots = post_rots[batch_idx, cam_idx].to(torch.float)
            post_trans = post_trans[batch_idx, cam_idx].to(torch.float)

        # B, N, _ = camera2lidar_trans.shape
        L = image_coords.shape[0]

        # undo post-transformation
        # B x N x L x 3
        if post_rots is not None:
            points = image_coords - post_trans.view(L, 3)
            points = torch.inverse(post_rots).view(L, 3, 3).matmul(points.unsqueeze(-1)).reshape(L, 3)
        else:
            points = image_coords

        # cam_to_lidar
        points = torch.cat((points[:, :2] * points[:, 2:3], points[:, 2:3]), -1)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(L, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points += camera2lidar_trans.view(L, 3)

        if "extra_rots" in kwargs and kwargs["extra_rots"] is not None:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots[batch_idx].view(L, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
            
        if "extra_trans" in kwargs and kwargs['extra_trans'] is not None:
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

            ret_dict[k]['pred_boxes'] = proposed_boxes[mask]#.cpu()
            ret_dict[k]['pred_scores'] = proposed_scores[mask]
            ret_dict[k]['pred_labels'] = proposed_labels[mask].int() # + 1 is done in preprocessed_detector

        return ret_dict 
