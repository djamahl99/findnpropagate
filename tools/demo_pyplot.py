import argparse
import time
from matplotlib.axes import Axes
import os
import cv2
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torchvision.utils import make_grid
from torchvision.ops import nms

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from typing import Dict, Tuple

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from pcdet.datasets import __all__ as ALL_DATASETS

from nuscenes.utils.geometry_utils import view_points

from pcdet.models.dense_heads.clip_box_cls_maskclip import CLIPBoxClassificationMaskCLIP
from pcdet.models.dense_heads import CLIPBoxClassification, GLIPBoxClassification
from pcdet.utils.box_utils import boxes_to_corners_3d

from torchvision.utils import save_image

from pcdet.models.preprocessed_detector import PreprocessedDetector, PreprocessedGLIP
from pcdet.utils.calibration_kitti import CalibrationTorch
from nuscenes.utils.data_classes import LidarPointCloud

CLASSES_NUSCENES_SEG = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

def hex_str_to_rgb(hex_str: str):
    assert len(hex_str) == 6
    
    rgb = []
    for i in range(3):
        rgb.append(int(hex_str[2*i:2*i+2], 16))

    return rgb

# PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
#             (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
#             (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
#             (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
#             (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
#             (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
#             (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
#             (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
#             (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
#             (134, 134, 103), (145, 148, 174), (255, 208, 186),
#             (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
#             (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
#             (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
#             (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
#             (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
#             (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
#             (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
#             (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
#             (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
#             (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
#             (191, 162, 208)]

# classname_to_color = {  # RGB.
#     # "noise": (0, 0, 0),  # Black.
#     # "animal": (70, 130, 180),  # Steelblue
#     "pedestrian": (0, 0, 230),  # Blue
#     # "human.pedestrian.child": (135, 206, 235),  # Skyblue,
#     # "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
#     # "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
#     # "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
#     # "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
#     # "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
#     "barrier": (112, 128, 144),  # Slategrey
#     "traffic_cone": (47, 79, 79),  # Darkslategrey
#     # "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
#     "bicycle": (220, 20, 60),  # Crimson
#     "bus": (255, 127, 80),  # Coral
#     # "bus": (255, 69, 0),  # Orangered
#     "car": (255, 158, 0),  # Orange
#     "construction_vehicle": (233, 150, 70),  # Darksalmon
#     "motorcycle": (255, 61, 99),  # Red
#     "trailer": (255, 140, 0),  # Darkorange
#     "truck": (255, 99, 71),  # Tomato
# }

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), hex_str_to_rgb('00BDE6'),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30)]
# # PALETTE = [(0, 0, 255)]

# PALETTE = ['227C9D', '17C3B2', 'FFCB77', '8B95C9', 'FE6D73', '0A2463', 'FB3640', '605F5E', '247BA0', '3F7D20']
# PALETTE = [hex_str_to_rgb(hex_str) for hex_str in PALETTE]
# PALETTE.append([0, 0, 0])


all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
knowns = ['car', 'bicycle', 'pedestrian']


# exit()
PALETTE = [[x/255 for x in y] for y in PALETTE]


all_colors = ['red', 'red', 'red', 'red', ]
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def render(axis: Axes,
            corners: np.ndarray,
            view: np.ndarray = np.eye(3),
            normalize: bool = False,
            colors: Tuple = ('b', 'r', 'k'),
            linewidth: float = 2) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    corners = view_points(corners, view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                    [corners.T[i][1], corners.T[i + 4][1]],
                    color=colors[2], linewidth=linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[0:2], axis=0)
    center_bottom = np.mean(corners.T[0:4], axis=0)
    # print('center_bottom, forward', center_bottom, center_bottom_forward)
    axis.plot([center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0], linewidth=linewidth)

class OWLViTDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        self.texts = [f"a photo of a {c}" for c in all_class_names]

    def forward(self, images):
        print('images ', images.device)
        # inputs = self.processor(text=self.texts, images=images, return_tensors="pt")
        # print('inputs', inputs)
        with torch.no_grad():
            outputs = self.model.image_guided_detection(pixel_values=images)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([images.size[::-1]])

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_image_guided_detection(outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries

        labels = []
        boxes = []
        scores = []
        idx = []
        for i, pred in enumerate(results):
            labels.append(pred['labels'])
            boxes.append(pred['boxes'])
            scores.append(pred['scores'])
            idx.append(torch.zeros((pred['labels'].shape[0])) + i)
        
        labels = torch.cat(labels, dim=0)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        idx = torch.cat(idx, dim=0)

        return boxes, labels, scores, idx

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

def draw_corners_on_image(corners, ax, color=(1, 1, 1), line_width=2, label='', max_num=500, tube_radius=None, x_offset=0, image_size=[900, 1600]):
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

    x2, y1 = corners[0, 1].max(), corners[0, 0].min()

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            ax.plot([start[1] + x_offset, end[1] + x_offset], [start[0], end[0]], color=color, lw=line_width)

            x2_, y1_ = max(start[1], end[1]), min(start[0], end[0])
            if x2_ > x2 and y1_ < y1:
                x2, y1 = x2_, y1_


        i, j = k + 4, (k + 1) % 4 + 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            ax.plot([start[1] + x_offset, end[1] + x_offset], [start[0], end[0]], color=color, lw=line_width)

            x2_, y1_ = max(start[1], end[1]), min(start[0], end[0])
            if x2_ > x2 and y1_ < y1:
                x2, y1 = x2_, y1_

        i, j = k, k + 4
        start, end = clip_line_to_image(corners[i], corners[j])
        if start is not None:
            ax.plot([start[1] + x_offset, end[1] + x_offset], [start[0], end[0]], color=color, lw=line_width)

            x2_, y1_ = max(start[1], end[1]), min(start[0], end[0])
            if x2_ > x2 and y1_ < y1:
                x2, y1 = x2_, y1_

    i, j = 0, 5
    start, end = clip_line_to_image(corners[i], corners[j])
    if start is not None:
        ax.plot([start[1] + x_offset, end[1] + x_offset], [start[0], end[0]], color=color, lw=line_width)

    i, j = 1, 4
    start, end = clip_line_to_image(corners[i], corners[j])
    if start is not None:
        ax.plot([start[1] + x_offset, end[1] + x_offset], [start[0], end[0]], color=color, lw=line_width)

    if label != '':
        # ax.text(corners[6, 1] + 5, corners[6, 0] + 5, label, color=color)
        ax.text(x2 + 5 + x_offset, y1 + 5, label, color=color)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/frustum_proposals_unknowns_1sweep.yaml',
                        help='specify the config for demo')
    # parser.add_argument('--ckpt', type=str, default='/home/uqdetche/OpenPCDet/output/nuscenes_models/transfusion_lidar_anchor_matching/default/ckpt/checkpoint_epoch_6.pth', help='specify the pretrained model')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--thresh', type=float, default=0.1, help='confidence thresh')
    parser.add_argument('--nms_thresh', type=float, default=0.1, help='nms iou thresh')
    parser.add_argument('--dist_thresh', type=float, default=3.0, help='remove boxes with dist of centre')
    parser.add_argument('--idx', type=int, default=None, help='idx')
    parser.add_argument('--find_class', type=int, default=None, help='idx')
    parser.add_argument('--save_blender', action='store_true', default=False, help='save blender')
    parser.add_argument('--knowns_only', action='store_true', default=False, help='knowns only')
    parser.add_argument('--unknowns_only', action='store_true', default=False, help='unknowns only')
    parser.add_argument('--two_colour', action='store_true', default=False, help='change to two colour, blue= knowns, pink= unknowns')
    parser.add_argument('--sweeps', type=int, default=10)

    parser.add_argument('--save_gt_crops', action='store_true', default=False, help='save crops')
    parser.add_argument('--training', action='store_true', default=False, help='training split')
    parser.add_argument('--vis_seg', action='store_true', default=False, help='visualise maskclip')
    parser.add_argument('--vis_glip', action='store_true', default=False, help='visualise glip')
    parser.add_argument('--gt_as_preds', action='store_true', default=False, help='use gt as preds for vis masks / clip crops')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

class Visualiser(nn.Module):
    def __init__(self, image_size=[900, 1600], class_names=None, 
            image_name='images_joined_pred.png', bev_name='bev_pred.png', 
            detector_jsons=None, show_pts_on_imgs=False, box_fmt='xywh', clip_cls_error_analysis=False) -> None:
        super().__init__()

        self.clip_cls_error_analysis = clip_cls_error_analysis
        self.box_fmt = box_fmt
        self.show_pts_on_imgs = show_pts_on_imgs
        self.image_name = image_name
        self.bev_name = bev_name
        self.class_names = class_names if class_names is not None else all_class_names

        self.image_order = [2, 0, 1, 5, 3, 4]
        # self.image_size = [512, 800]
        self.image_size = image_size if image_size is not None else [900, 1600]

        self.num_sample_pts = 100


        # self.detector = TorchVisionDetector().to('cuda')
        # self.detector = OWLViTDetector().to('cuda')

        self.detector = None
        if detector_jsons is not None:
            self.detector = PreprocessedDetector(detector_jsons, class_names=class_names)

        # self.detector = PreprocessedGLIP()
        # self.box_fmt = 'xyxy'
        # self.detector = PreprocessedDetector(['glip_train_preds.json'], class_names=class_names)


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

    def project_to_camera_kitti(self, points, calib: CalibrationTorch):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()
        
        cur_coords, cur_depth = calib.lidar_to_img(cur_coords[..., :3])
        cur_coords = torch.cat((cur_coords, cur_depth.unsqueeze(1)), dim=-1)

        on_img = torch.ones(cur_coords.shape[:-1], dtype=torch.bool)

        return cur_coords, on_img

    def get_geometry_at_image_coords_kitti(self, image_coords, calib: CalibrationTorch):

        pts_rect = calib.img_to_rect(image_coords[..., 0], image_coords[..., 1], image_coords[..., 2])

        return calib.rect_to_lidar(pts_rect)

    def forward(self, batch_dict, pred_dicts, vis_images=None, image_name=None):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        if image_name is not None:
            self.image_name = image_name

        if self.image_size is None or 'camera_imgs' not in batch_dict:
            if 'images' in batch_dict:
                self.kitti_cam_forward(batch_dict, pred_dicts)

            return self.bev_forward(batch_dict, pred_dicts)

        self.nusc_cam_forward(batch_dict, pred_dicts, vis_images)

    def kitti_cam_forward(self, batch_dict, pred_dicts):
        images = batch_dict['images']

        # save_image(images, 'kitti_imgage.png')


        batch_size = batch_dict['batch_size']

        N, pred_dim = pred_dicts[0]['pred_boxes'].shape[-2:]

        pred_boxes = pred_dicts[0]['pred_boxes'].reshape(-1, pred_dim)
        # print('pred_boxes', pred_dicts[0]['pred_boxes'].shape, pred_dicts)
        
        corners = boxes_to_corners_3d(pred_boxes)
        corners = corners.reshape(1, -1, 3)

        pred_scores = pred_dicts[0]['pred_scores'].clone()
        pred_labels = pred_dicts[0]['pred_labels'].clone().long()

        # box number
        box_idx = torch.arange(0, N).reshape(1, N, 1).repeat(1, 1, 8).reshape(1, N*8)
        coord_labels = pred_labels.reshape(1, N, 1).repeat(1, 1, 8).reshape(1, N*8)
        theta = torch.eye(2, 3).unsqueeze(0)
        unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]) # -1, 1 grid
        # print('box idx', box_idx)
        # print('images', images.shape, images.dtype, images.device)

        # det_boxes, det_labels, det_scores, det_idx = self.detector(images.reshape(-1, 3, images.shape[-2], images.shape[-1]))
        # print('det_boxes', det_boxes.shape)
        
        if self.detector is not None:
            # 2d multiview detections loaded (loaded from coco jsons)
            det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.detector(batch_dict)

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]
            cur_box_coords = corners[b, :, :3]
            cur_idx = box_idx[b]
            cur_labels = coord_labels[b]
            cur_images = images[b]

            calib = batch_dict['calib'][b]
            calib = CalibrationTorch(calib, device=cur_points.device)

            if self.detector is not None:
                detector_batch_mask = (det_batch_idx == b)
                cur_boxes, cur_det_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            # all in one image
            images_joined = cur_images


            hj, wj = images_joined.shape[-2:]
            images_joined_np = images_joined.cpu().permute(1, 2, 0).numpy()

            dpi = 50
            # fig, ax = plt.subplots()
            fig = plt.figure(frameon=False, dpi=dpi)
            fig.set_size_inches(wj/10, hj/10)
            # print('wj', wj, hj)
            # fig.set_size
            ax = fig.gca()
            ax.imshow(images_joined_np)
            plt.axis('off')

            curr_x_off = 0

            proj_box_coords, box_on_img = self.project_to_camera_kitti(cur_box_coords, calib)
            cam_pts, pts_on_img = self.project_to_camera_kitti(cur_points, calib)

            all_coords = proj_box_coords.long().cpu()

            # get boxes with at least one corner on the current image
            masked_idx = cur_idx[box_on_img]

            if self.show_pts_on_imgs:
                all_coords_img = cam_pts[pts_on_img].cpu().numpy()
                depths = cam_pts[pts_on_img, ..., 2].cpu().numpy()
                ax.scatter(all_coords_img[..., 0] + curr_x_off, all_coords_img[..., 1], c=depths, cmap='jet')

            if self.detector is not None:
                cam_boxes, cam_labels, cam_scores = cur_boxes, cur_det_labels, cur_scores
                for i in range(cam_labels.shape[0]):
                    lbl_idx = cam_labels[i].item() - 1
                    # label_txt = f"{coco_classes[lbl_idx.item()]} {cur_det_scores[i]:.2f}"
                    label_txt = f"predict {lbl_idx.item()} {cam_scores[i]:.2f}"
                    # draw_corners_on_image(box_coords, ax, color=PALETTE[lbl_idx.item()], label=label_txt)

                    if self.box_fmt == 'xywh':
                        x1, y1, w, h = cam_boxes[i].cpu()
                        x2, y2 = x1 + w, y1 + h
                    else:
                        x1, y1, x2, y2 = cam_boxes[i].cpu()

                    # ax.text(x2 + 5 + curr_x_off, y1 + 5, label_txt, color=PALETTE[lbl_idx.item() % len(PALETTE)])

                    ax.add_patch(Rectangle(xy=[x1+curr_x_off, y1], width=x2-x1, height=y2-y1, color=PALETTE[lbl_idx.item() % len(PALETTE)], fill=False, linewidth=7, linestyle='--'))

            last_idx = -1
            for i, idx in enumerate(masked_idx):
                if idx == last_idx:
                    continue
                last_idx = idx

                coord_mask = (cur_idx == idx)
                box_coords = all_coords[coord_mask].clone().cpu()

                on_cam = (box_coords[..., 2] >= 0).all()
                if not on_cam:
                    print('not on cam!', box_coords)
                    continue

                box_coords_ = box_coords.clone()

                print('box_coords_', box_coords_.min(dim=0).values, box_coords_.max(dim=0).values, (wj, hj))
                # clamp to image
                box_coords[..., 0] = torch.clamp(box_coords_[..., 1].clone(), 0, hj)
                box_coords[..., 1] = torch.clamp(box_coords_[..., 0].clone(), 0, wj)

                # box_coords = self.clip_to_image(box_coords)
                # print('box_coords', box_coords)

                ax.scatter(box_coords[..., 1].numpy(), box_coords[..., 0].numpy())
                # box_coords = box_coords[..., [1, 0]]

                # lbl_idx = masked_labels[coord_mask][0]
                lbl_idx = pred_labels[idx] - 1
                score = pred_dicts[0]['pred_scores'][idx]

                # label_txt = f"BOX {idx.item()}"
                label_txt = f"{self.class_names[lbl_idx.item()]} {score.item():.2f}"
                
                draw_corners_on_image(box_coords, ax, color=PALETTE[lbl_idx.item() % len(PALETTE)], label=label_txt, line_width=4, image_size=[hj, wj])

            # plt.savefig(f'images_joined_pred.png', bbox_inches='tight', dpi=100)
            plt.savefig(self.image_name, bbox_inches='tight', dpi=dpi)
            plt.close()

    def nusc_cam_forward(self, batch_dict, pred_dicts, vis_images=None):
        # x = batch_dict['image_fpn'] 
        # print("batch dict", batch_dict.keys())
        # print("preds_dict", [x.keys() for x in preds_dict])
        images = batch_dict['camera_imgs']

        batch_size = batch_dict['batch_size']

        N, pred_dim = pred_dicts[0]['pred_boxes'].shape[-2:]

        pred_boxes = pred_dicts[0]['pred_boxes'].reshape(-1, pred_dim)
        # print('pred_boxes', pred_dicts[0]['pred_boxes'].shape, pred_dicts)
        
        corners = boxes_to_corners_3d(pred_boxes)
        corners = corners.reshape(1, -1, 3)

        pred_scores = pred_dicts[0]['pred_scores'].clone()
        pred_labels = pred_dicts[0]['pred_labels'].clone().long()

        # box number
        box_idx = torch.arange(0, N).reshape(1, N, 1).repeat(1, 1, 8).reshape(1, N*8)
        coord_labels = pred_labels.reshape(1, N, 1).repeat(1, 1, 8).reshape(1, N*8)
        theta = torch.eye(2, 3).unsqueeze(0)
        unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]) # -1, 1 grid
        # print('box idx', box_idx)

        # print('images', images.shape, images.dtype, images.device)

        # det_boxes, det_labels, det_scores, det_idx = self.detector(images.reshape(-1, 3, images.shape[-2], images.shape[-1]))
        # print('det_boxes', det_boxes.shape)
        
        if self.detector is not None:
            # 2d multiview detections loaded (loaded from coco jsons)
            det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.detector(batch_dict)

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]
            cur_box_coords = corners[b, :, :3]
            cur_idx = box_idx[b]
            cur_labels = coord_labels[b]
            cur_images = images[b]

            if self.detector is not None:
                detector_batch_mask = (det_batch_idx == b)
                print('det_batch_idx', det_batch_idx)
                # print('det_boxes, det_labels, det_scores, det_cam_idx', det_boxes.shape, det_labels.shape, det_scores.shape, det_cam_idx.shape)
                # cur_boxes, cur_det_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]


            # all in one image
            if vis_images is None:
                images_joined = make_grid(cur_images[self.image_order], nrow=len(self.image_order), padding=0, normalize=True, scale_each=True)
            else:
                images_joined = make_grid(vis_images, nrow=len(self.image_order), padding=0, normalize=True, scale_each=True)

            # print('images_joined', images_joined.shape, vis_images.shape if vis_images is not None else 'no vis')

            wj, hj = images_joined.shape[-2:]
            images_joined_np = images_joined.cpu().permute(1, 2, 0).numpy()
            images_joined_cv = images_joined_np.copy()
            images_joined_cv = (images_joined_cv - images_joined_cv.min()) / (images_joined_cv.max() - images_joined_cv.min())

            images_joined_cv = (images_joined_cv *255).astype(np.uint8)

            cv2.imwrite(self.image_name.replace('.png', '.raw.png'), images_joined_cv[:, :, [2, 1, 0]])


            dpi = 10
            # fig, ax = plt.subplots()
            fig = plt.figure(frameon=False)
            fig.set_size_inches(16*6, 9)
            print('wj', wj, hj)
            # fig.set_size
            ax = fig.gca()
            ax.imshow(images_joined_np)
            plt.axis('off')

            curr_x_off = 0

            for c in self.image_order:
                proj_box_coords, box_on_img = self.project_to_camera(batch_dict, cur_box_coords, b, cam_idx=c)
                cam_pts, pts_on_img = self.project_to_camera(batch_dict, cur_points, b, cam_idx=c)


                # masked_coords = cur_coords[c, on_img[c]].long().cpu().numpy()
                all_coords = proj_box_coords[0, :].long().cpu()

                # get boxes with at least one corner on the current image
                masked_idx = cur_idx[box_on_img[0]]
                # masked_labels = cur_labels[box_on_img[0]]

                if self.show_pts_on_imgs:
                    all_coords_img = cam_pts[0, pts_on_img[0]].cpu().numpy()
                    depths = cam_pts[0, pts_on_img[0], ..., 2].cpu().numpy()
                    ax.scatter(all_coords_img[..., 0] + curr_x_off, all_coords_img[..., 1], c=depths, cmap='jet')

                if self.detector is not None:
                    box_cam_mask = (det_cam_idx == c)
                    cam_boxes, cam_labels, cam_scores = det_boxes[box_cam_mask], det_labels[box_cam_mask], det_scores[box_cam_mask]
                    print('cam_boxes, ...', cam_boxes.shape, cam_labels.shape, cam_scores.shape, det_cam_idx.shape) 

                # cur_boxes, cur_det_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]


                    if self.box_fmt == 'xywh':
                        xy = cam_boxes[..., :2]
                        wh = cam_boxes[..., 2:]
                        xy2 = xy + wh
                        cam_boxes = torch.cat((xy, xy2), dim=-1)
                    nms_indices = nms(cam_boxes, cam_scores, 0.3)
                    cam_boxes = cam_boxes[nms_indices]
                    cam_labels = cam_labels[nms_indices]
                    cam_scores = cam_scores[nms_indices]

                    score_mask = (cam_scores > 0.3)
                    cam_boxes = cam_boxes[score_mask]
                    cam_labels = cam_labels[score_mask]
                    cam_scores = cam_scores[score_mask]

                    for i in range(cam_labels.shape[0]):
                        lbl_idx = cam_labels[i] - 1
                        # label_txt = f"{coco_classes[lbl_idx.item()]} {cur_det_scores[i]:.2f}"
                        # label_txt = f"predict {self.class_names[lbl_idx.item()]} {cam_scores[i]:.2f}"
                        # draw_corners_on_image(box_coords, ax, color=PALETTE[lbl_idx.item()], label=label_txt)

                        label_txt = f'{self.class_names[lbl_idx.item()]} {cam_scores[i]:.2f}'

                        print('labe', label_txt)

                        # if self.box_fmt == 'xywh':
                        #     x1, y1, w, h = cam_boxes[i].cpu()
                        #     x2, y2 = x1 + w, y1 + h
                        # else:
                        #     x1, y1, x2, y2 = cam_boxes[i].cpu()

                        x1, y1, x2, y2 = cam_boxes[i].cpu()


                        ax.text(x2 + 5 + curr_x_off, y1 + 5, label_txt, color=PALETTE[lbl_idx.item() % len(PALETTE)])

                        ax.add_patch(Rectangle(xy=[x1+curr_x_off, y1], width=x2-x1, height=y2-y1, color=PALETTE[lbl_idx.item() % len(PALETTE)], fill=False, linewidth=5, linestyle='--'))
                        images_joined_cv = cv2.rectangle(images_joined_cv, (int(x1 + curr_x_off), int(y1)), (int(x2 + curr_x_off), int(y2)), tuple(int(x*255) for x in PALETTE[lbl_idx.item() % len(PALETTE)]), thickness=5)

                last_idx = -1
                for i, idx in enumerate(masked_idx):
                    if idx == last_idx:
                        continue
                    last_idx = idx

                    coord_mask = (cur_idx == idx)
                    box_coords = all_coords[coord_mask].clone().cpu()

                    # box_coords = self.clip_to_image(box_coords[..., [1, 0]])
                    box_coords = box_coords[..., [1, 0]]

                    # box_coords[..., 1] += curr_x_off

                    # lbl_idx = masked_labels[coord_mask][0]
                    lbl_idx = pred_labels[idx] - 1
                    score = pred_dicts[0]['pred_scores'][idx]

                    # label_txt = f"BOX {idx.item()}"
                    # label_txt = f"{self.class_names[lbl_idx.item()]} {score.item():.2f}"
                    # label_txt = None
                    label_txt = f'{score.item():.2f}'
                    
                    # draw_corners_on_image(box_coords, ax, color=PALETTE[lbl_idx.item() % len(PALETTE)], label=label_txt, line_width=4, x_offset=curr_x_off)
                    images_joined_cv = draw_corners_on_cv(box_coords, images_joined_cv, color=PALETTE[lbl_idx.item() % len(PALETTE)], line_width=4, x_offset=curr_x_off)

                curr_x_off += self.image_size[1]

            plt.savefig(self.image_name, bbox_inches='tight')
            plt.close()

            # print('final', images_joined_cv.min(), images_joined_cv.max(), images_joined_cv.shape)
            cv2.imwrite(self.image_name.replace('.png', '.cv2.png'), images_joined_cv[:, :, [2, 1, 0]])

            for c in range(6):
                print('images_joined_cv', images_joined_cv.shape)
                print('c', c, int(1600*c), int(1600*(c+1)))
                # print('c', images_joined_cv[int(1600*c):int(1600*(c+1)), :, [2, 1, 0]].shape)
                cv2.imwrite(self.image_name.replace('.png', f'.c{c}.png'), images_joined_cv[:, int(1600*c):int(1600*(c+1)), [2, 1, 0]])

            self.bev_forward(batch_dict, pred_dicts)

    def bev_forward(self, batch_dict, pred_dicts):
        if 'calib' in batch_dict:
            print("batch dict", batch_dict.keys())
            print('calib', batch_dict['calib'])
            print('frame_id', batch_dict['frame_id'])
                # if 'road_plane' in batch_dict:
                # print('road_plane', batch_dict['road_plane'])

        # def recalc_rot(points, boxes):
        #     new_boxes = boxes.clone()
        #     points_rel = points.clone().reshape(1, -1, 3).repeat(boxes.shape[0], 1, 1)

        #     # centre
        #     points_rel = points_rel - boxes[:, None, :3]

        #     # rotate
        #     points_rot = common_utils.rotate_points_along_z(points_rel, -boxes[..., 6])
        #     print('points_rot', points_rot.shape)
        #     point_masks = (points_rot[..., :3].abs() < boxes[:, None, 3:6] / 2.0).all(dim=2)

        #     for i in range(boxes.shape[0]):
        #         curr_box = boxes[i]
        #         box_point_mask = point_masks[i]

        #         curr_points_rel = points_rel[i, box_point_mask]

        #         # svd
        #         _, _, Vh = torch.linalg.svd(curr_points_rel)

        #         Vh = Vh / Vh.norm(dim=1, keepdim=True)


        #         # assume default angle is somewhat correct
        #         orig_vecs = torch.stack([torch.cos(curr_box[..., 6]), torch.sin(curr_box[..., 6]), torch.zeros_like(curr_box[..., 6])])
        #         print('orig_vecs', orig_vecs.shape)

        #         svd_dots = Vh @ orig_vecs
                
        #         # best_idx = torch.argmax(svd_dots.abs(), dim=0)
        #         best_idx = 0

        #         chosen_V = Vh[best_idx]

        #         new_boxes[i, 6] = torch.atan2(chosen_V[1], chosen_V[0])

        #         # if svd_dots[best_idx] < 0: # negative
        #             # new_boxes[i, 6] += torch.pi

        #         print('old dir', boxes[i, 6])
        #         print('new dir', new_boxes[i, 6])

        #     return new_boxes

        # birds eye view
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        # fig.set_size_inches(5,5)

        B = batch_dict['batch_size']

        points = batch_dict['points'][..., 1:4].cpu().reshape(-1, 3).permute(1, 0)

        pred_dim = pred_dicts[0]['pred_boxes'].shape[-1]
        pred_boxes = pred_dicts[0]['pred_boxes'].reshape(-1, pred_dim).cpu()

        # new_preds = recalc_rot(points, pred_boxes)
        # ncorners = boxes_to_corners_3d(new_preds)
        # ncorners = ncorners.reshape(1, -1, 3)

        corners = boxes_to_corners_3d(pred_boxes)
        corners = corners.reshape(1, -1, 3)

        # pred_scores = pred_dicts[0]['pred_scores'].clone()
        # pred_labels = pred_dicts[0]['pred_labels'].clone().long()
        
        B, gN = batch_dict['gt_boxes'].shape[0:2]

        gt_boxes = batch_dict['gt_boxes'].reshape(B*gN, -1)
        gt_labels = gt_boxes[..., -1].long()

        # ego_vehicle = torch.tensor([[0, -1.0, (-5.0 + 3.0)/2.0, 5.0, 3.0, 8.0, np.pi/2.0]], dtype=torch.float32, device=gt_boxes.device)

        # gt_boxes = torch.cat((gt_boxes[:, :7], ego_vehicle), dim=0)
        # gt_labels = torch.cat((gt_labels, gt_labels.new_zeros((1))))
        # gN += 1

        corners_gt = boxes_to_corners_3d(gt_boxes)
        corners_gt = corners_gt.reshape(B*gN, 8, 3)

        pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        eval_range = max(pc_range)

        points = view_points(points, np.eye(4), normalize=False)
        dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / eval_range)
        ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

        # clip2scene seg vis (bus)
                # PRED_DIR = "/media/bigdata/uqdetche/clip2scene_preds/"
                # LIDAR_DIR = "/media/bigdata/uqyluo/3D_DA/NUSCENES/v1.0-trainval/samples/LIDAR_TOP/"
                # clip2scene_path = PRED_DIR + batch_dict['frame_id'][0] + ".bin.pth"
                # lidar_path = LIDAR_DIR + batch_dict['frame_id'][0] + '.bin'

                # assert os.path.exists(clip2scene_path), f'{clip2scene_path} does not exist!'

                # preds = torch.load(clip2scene_path, map_location='cpu')
                # pc_original = LidarPointCloud.from_file(lidar_path)
                # points = pc_original.points.T[:, :3]
                # pred_labels = preds['predictions']
                # points = view_points(points.T, np.eye(4), normalize=False)
                # # dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
                # # colors = np.minimum(1, dists / eval_range)
                # pc_range = [-2.5, 15, 0, 12.5, 35, 0]
                # # ax.set_xlim(0, 20)
                # # ax.set_ylim(15, 40)

                # bg_mask = np.zeros_like(pred_labels, dtype='bool')
                # bus_mask = None

                # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                # for i, cls_name in enumerate(CLASSES_NUSCENES_SEG):
                #     point_mask = (pred_labels == (i + 1))
                #     # if cls_name not in det_classes:
                #         # continue
                #     if cls_name != 'bus':
                #     #     bg_mask = np.bitwise_or(bg_mask, point_mask)
                #         continue
                #     else:
                #         bus_mask = point_mask
                #     cls_points = points[:, point_mask]
                #     # color = PALETTE[i]
                #     # color = 'black' if cls_name not in all_class_names else colors[all_class_names.index(cls_name)]
                #     # ax.scatter(cls_points[0, :], cls_points[1, :], color='red', s=0.2, zorder=10, label='Bus')
                #     # ax.scatter(cls_points[0, :], cls_points[1, :], color=color, s=0.2, zorder=10, label=cls_name)
                #     ax.scatter(cls_points[0, :], cls_points[1, :], color='red', s=0.2, zorder=10, label='Bus')

                # bg_points = points[:, ~bus_mask]
                # ax.scatter(bg_points[0, :], bg_points[1, :], color=(0, 0, 0), s=0.2, label='other')
                # lgnd = ax.legend(title='CLIP2Scene Label', loc='upper left')
                # for i in range(len(lgnd.legend_handles)):
                #     lgnd.legend_handles[i]._sizes = [30]


        ax.plot(0, 0, 'x', color='black')

        if not self.clip_cls_error_analysis:
            for i, gt_box in enumerate(corners_gt):
                lbl = gt_labels[i].item() -1
                
                color = PALETTE[lbl % len(PALETTE)]
                render(ax, gt_box.cpu().numpy().T, colors=('g', 'g', 'g'), linewidth=3)

            for pred_box in corners.reshape(-1, 8, 3):
                render(ax, pred_box.cpu().numpy().T, colors=('b', 'b', 'b'), linewidth=1)
        else:
            pred_labels = pred_dicts[0]['pred_labels'].clone().long()
            orig_labels = pred_dicts[0]['orig_labels'].clone().long()

            red_patch = mpatches.Patch(color='red', label='Incorrect')
            green_patch = mpatches.Patch(color='green', label='Correct')
            plt.legend(handles=[green_patch, red_patch], title='CLIP Classification')

            for pred_box, orig_label, pred_label in zip(corners.reshape(-1, 8, 3), orig_labels, pred_labels):
                # print('orig_label', orig_label, 'pred_label', pred_label)
                color = 'green' if orig_label.item() == pred_label.item() else 'red'
                render(ax, pred_box.cpu().numpy().T, colors=(color, color, color), linewidth=3)

        # for pred_box in ncorners.reshape(-1, 8, 3):
            # render(ax, pred_box.cpu().numpy().T, colors=('purple', 'purple', 'purple'), linewidth=1)

            # draw_corners_on_image(pred_box.cpu(), ax, color=(0, 1, 0))

        # for i, pred_box in enumerate(pred_boxes):
        #     print('pred_box', pred_box)
        #     color = PALETTE[i % len(PALETTE)]
        #     ax.scatter(pred_box[..., 1].cpu(), pred_box[..., 0].cpu(), color=color)

        # ax.scatter(gt_boxes[..., 0], gt_boxes[..., 1], color=(0, 0, 1), label='gt boxes', s=100)
        # ax.scatter(query_pos[..., 0].cpu(), query_pos[..., 1].cpu(), color=(1, 0, 0), label='2d box ->3d queries')
        # ax.scatter(res_layer["center"][..., 0].detach().cpu(), res_layer["center"][..., 1].detach().cpu(), color=(0, 1, 0), label='predicted centres')
        # plt.legend()
        # Limit visible range.
        # axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        plt.savefig(self.bev_name, bbox_inches='tight', dpi=100)
        plt.close()

def kmeans_filter(boxes, scores, k=20, steps=10):
    # assume same label
    if boxes.shape[0] <= k:
        return torch.ones((scores.shape), dtype=torch.bool, device=boxes.device)

    centres = boxes[torch.topk(scores, k=k).indices, :3]

    for step in range(steps):
        # assign to centres
        dists = torch.cdist(boxes[:, :3], centres)
        assigns = torch.argmin(dists, dim=1)

        print('num assigned', [int((assigns == i).sum()) for i in range(k)])

        # update centres (score weighted)
        centres = []
        for l in range(k):
            assign_mask = (assigns == l)

            if assign_mask.sum() == 0:
                centres.append(boxes[0, :3]*0)
                continue

            assign_scores = scores[assign_mask]
            assign_boxes = boxes[assign_mask]

            assign_weights = (assign_scores) / (assign_scores.sum() + 1e-7)
            centre = assign_boxes[:, :3] * assign_weights.reshape(-1, 1)
            centre = centre.sum(dim=0)

            centres.append(centre)
        
        centres = torch.stack(centres)

    # return the most confident in each cluster
    return_mask = torch.zeros((scores.shape), dtype=torch.bool, device=boxes.device)
    dists = torch.cdist(boxes[:, :3], centres)
    assigns = torch.argmin(dists, dim=1)

    for l in range(k):
        best_score = scores[assigns == l].max()
        best_mask = (assigns == l) & (scores >= best_score)

        return_mask[best_mask] = True

    return return_mask
        
def save_blender_vis(batch_dict, pred_dicts, color_by='gt', knowns_only=False):
    known_ids = [all_class_names.index(x) + 1 for x in knowns]
    print('known ids', known_ids)

    def color_pcl(cur_points, boxes, labels):
        pt_colors = torch.zeros_like(cur_points)

        point_box_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points.unsqueeze(0), boxes[..., :7].reshape(1, -1, 7))
        point_box_indices = point_box_indices.reshape(-1)

        for i in range(labels.shape[0]):
            pt_mask = (point_box_indices == i)
            color = torch.tensor([float(x) for x in PALETTE[labels[i].item() - 1]], device=pt_colors.device)
            pt_colors[pt_mask] = color.reshape(1, 3)

        return pt_colors

    # grid = cam_points[..., :2].clone().reshape(1, 1, -1, 2)
    cur_points = batch_dict['points'][..., 1:4]

    pred_boxes = pred_dicts[0]['pred_boxes']
    pred_labels = pred_dicts[0]['pred_labels']

    if knowns_only:
        pred_mask = torch.zeros_like(pred_labels, dtype=torch.bool)
        for i in range(len(pred_labels)):
            pred_mask[i] = pred_labels[i] in known_ids

        
        # filter
        for k in ['pred_labels', 'pred_boxes', 'pred_scores']:
            pred_dicts[0][k] = pred_dicts[0][k][pred_mask]
        pred_boxes = pred_dicts[0]['pred_boxes']
        pred_labels = pred_dicts[0]['pred_labels']

    if 'gt_boxes' in batch_dict:
        gt_boxes = batch_dict['gt_boxes']
        box_dim = gt_boxes.shape[-1]
        gt_boxes = gt_boxes.reshape(-1, box_dim)

        gt_labels = gt_boxes[..., -1].long()

        if knowns_only:
            gt_mask = torch.zeros_like(gt_labels, dtype=torch.bool)
            for i in range(len(gt_labels)):
                gt_mask[i] = gt_labels[i] in known_ids

            print('knowns', pred_mask, pred_labels)
                
            gt_boxes = gt_boxes[gt_mask]
            gt_labels = gt_labels[gt_mask]


        gt_boxes_np = gt_boxes.cpu().numpy()

        np.save('nusc_vis_gt_boxes', gt_boxes_np)

    if color_by == 'gt':
        pt_colors = color_pcl(cur_points, gt_boxes, gt_labels)
    else:
        pt_colors = color_pcl(cur_points, pred_boxes, pred_labels)

    pred_boxes = torch.cat((pred_boxes.cpu(), pred_labels.unsqueeze(-1).cpu()), dim=-1)

    np.save('nusc_vis_col', pt_colors.cpu().numpy())
    np.save('nusc_vis_pcl', cur_points.cpu().numpy())
    np.save('nusc_vis_pred_boxes', pred_boxes.cpu().numpy())
def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    print(cfg.CLASS_NAMES)

    # print('cfg.DATA_CONFIG', cfg.DATA_CONFIG)

    if cfg.DATA_CONFIG.DATASET.lower() == 'nuscenes':
        print('sweeps?', cfg.DATA_CONFIG.MAX_SWEEPS)


        cfg.DATA_CONFIG.MAX_SWEEPS = args.sweeps
    dataset_cfg = cfg.DATA_CONFIG
    dataset_cfg['DATA_AUGMENTOR']['AUG_CONFIG_LIST'] = []

    demo_dataset = ALL_DATASETS[dataset_cfg.DATASET](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        # root_path=dataset_cfg.DATA_PATH, # none -> use cfg
        training=args.training,
        logger=logger,
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    # print('cfg.DATA_CONFIG', cfg.DATA_CONFIG)

    image_size = None
    if 'CAMERA_CONFIG' in cfg.DATA_CONFIG:
        image_size = cfg.DATA_CONFIG.CAMERA_CONFIG.IMAGE.FINAL_DIM

    # visualiser = Visualiser(image_size, class_names=cfg.CLASS_NAMES, detector_jsons=['/home/uqdetche/GLIP/GLIP_MMDET.json'])
    # preds_path = '/home/uqdetche/OpenPCDet/tools/coco_val2_'
    # preds_path = '/home/uqdetche/GLIP/OWL_'
    preds_path = '/home/uqdetche/GLIP/jsons/OWL_'
    camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

    if not args.vis_glip:
        preds_paths = None
    else: 
        preds_paths = ['/home/uqdetche/GLIP/GLIP_MMDET.json']

    print('preds_paths', preds_paths)

    visualiser = Visualiser(image_size, class_names=cfg.CLASS_NAMES, detector_jsons=preds_paths, box_fmt='xywh', clip_cls_error_analysis=False)
    # visualiser = Visualiser(image_size, class_names=cfg.CLASS_NAMES)

    clip_post_classifier = None

    if args.vis_seg:
        clip_post_classifier = CLIPBoxClassificationMaskCLIP().cuda() # 2seconds
    # clip_post_classifier = CLIPBoxClassification().cuda() #  1.3 seconds
    # if args.vis_glip:
        # clip_post_classifier = GLIPBoxClassification(preds_paths, cfg.CLASS_NAMES, image_size=image_size)

    if not args.gt_as_preds:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        if args.ckpt is not None:
            model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

    # where to start searching for unknowns
    # idx = 600
    # idx = 138
    # idx = 400
    idx = 0
    # idx = 790
    # idx = 970
    # idx = 1521
    # idx = 1871
    # idx = 2140
    # idx = 3620
    # idx = 4190
    # idx = 4850
    # idx = 5120

    idx = 761

    num_unk_thresh = 6

    if args.idx is None:
        labels_to_find = [2, 4, 7, 10]
        # label_to_find = 7 # motorcycle = 7
        # label_to_find = 2 # cycle = 2 (kitti)
        # for idx, data_dict in enumerate(demo_dataset):
        for i in range(idx//10 + 10, len(demo_dataset) // 10):
            idx = i * 10
            data_dict = demo_dataset[idx]
            gt_boxes = data_dict['gt_boxes']
            gt_boxes_labels = gt_boxes[..., -1].astype(np.int32)

            # print('labels', gt_boxes_labels)
            # labels_found = [x in gt_boxes_labels for x in labels_to_find]

            labels_found = [(gt_boxes_labels == x).sum() for x in labels_to_find]
            
            print('labels found', labels_found, sum(labels_found))
            
            if sum(labels_found) > num_unk_thresh:
                break

            # if all([x > 0 for x in labels_found]) or sum(labels_found) > num_unk_thresh:
                # break
            # for label_to_find in labels_to_find:
            #     if label_to_find in gt_boxes_labels:
            #         break


    if args.idx is not None:
        idx = args.idx

    print('idx', idx)

    with torch.no_grad():
        # for idx, data_dict in enumerate(demo_dataset):
        data_dict = demo_dataset[idx]

        print('frame_id', data_dict['frame_id'])

        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        if args.knowns_only:
            known_ids = [all_class_names.index(x) + 1 for x in knowns]

            # filter batch dict
            labels = data_dict['gt_boxes'][..., -1].long()
            labels_shape = labels.shape
            labels = labels.reshape(-1)

            known_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx, lbl in enumerate(labels):
                known_mask[idx] = lbl in known_ids
            
            known_mask = known_mask.reshape(labels_shape)
            
            print('shape', known_mask.shape, data_dict['gt_boxes'].shape)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][known_mask].unsqueeze(0)
            print('after', data_dict['gt_boxes'].shape)
            

        if not args.gt_as_preds:
            pred_dicts, _ = model.forward(data_dict)

            if 'pseudo_samples_mask' in data_dict:
                print('pseudo_samples_mask', data_dict['pseudo_samples_mask'])

            # frame_id = data_dict['frame_id'][0]
            # print('frame', frame_id)

            # path = f'../data/pseudo_labels/self_training_transfusion/' + f"{frame_id.replace('.', '_')}.pth"
            # pred_dict = torch.load(path, map_location='cuda')
            # pred_dicts = pred_dict

            # FILTERING ######################################################
            pred_dict = pred_dicts[0]
            pred_boxes = pred_dict['pred_boxes']
            pred_scores = pred_dict['pred_scores']
            pred_labels = pred_dict['pred_labels']

            if pred_scores.numel() > 0 and not visualiser.clip_cls_error_analysis:

                # print('pred_dict', pred_dict)

                pred_scores = pred_scores.to(pred_boxes.device)
                print('pred_scores', pred_scores.min(), pred_scores.max())

                nms_indices, _ = iou3d_nms_utils.nms_normal_gpu(pred_boxes[:, :7],
                    pred_scores, thresh=args.nms_thresh)

                nms_mask = torch.zeros((pred_scores.shape), dtype=torch.bool, device=pred_scores.device)
                nms_mask[nms_indices] = True

                print('nms removed', (~nms_mask).sum())

                # pred_mask = (pred_scores >= args.thresh) & (torch.norm(pred_boxes[..., 0:3], dim=-1) > args.dist_thresh) & nms_mask
                pred_mask = (pred_scores >= args.thresh) & nms_mask
                pred_boxes = pred_boxes[pred_mask]
                pred_labels = pred_labels[pred_mask]
                pred_scores = pred_scores[pred_mask]

                pred_dict = dict(pred_boxes=pred_boxes, pred_scores=pred_scores, pred_labels=pred_labels)
                pred_dicts = [pred_dict]
            # FILTERING ######################################################

            # if clip_post_classifier is not None:
            #     start = time.time()

            #     pred_dicts = clip_post_classifier(data_dict, pred_dicts)
            #     end = time.time()

            #     print('clip took', (end-start), 'seconds')


        else:
            pred_dicts = []

            for i in range(data_dict['batch_size']):
                b_gt_boxes = data_dict['gt_boxes'][i]
                gt_labels = b_gt_boxes[..., -1].long()
                gt_boxes = b_gt_boxes[..., :7]

                is_empty = gt_labels > 10
                gt_labels = gt_labels[~is_empty]
                gt_boxes = gt_boxes[~is_empty]
                gt_scores = torch.ones_like(gt_labels, dtype=torch.float)

                pred_dicts.append(dict(pred_boxes=gt_boxes, pred_scores=gt_scores, pred_labels=gt_labels))


        if args.save_gt_crops:
            # save crops
            clip_post_classifier(data_dict, pred_dicts, keep_crops=True, relabel=False)
            
            crops = clip_post_classifier.crop_infos['crops']
            logits = clip_post_classifier.crop_infos['logits']

            logits_max = torch.max(logits, dim=1)
            order = torch.argsort(-logits_max.values)
            crops_folder = Path('paper_scripts/clip_crops_fig')
            crops_folder.mkdir(parents=True, exist_ok=True)

            for oidx, idx in enumerate(order):
                class_name = clip_post_classifier.all_class_names[logits_max.indices[idx].item()]

                # with open(crops_folder / f'info_{oidx}.txt', 'w') as f:
                #     f.write(f'predict {class_name} with {logits_max.values[idx].item():.3f}')
                save_image(crops[idx], crops_folder / f'crops_{oidx}_{class_name}_{logits_max.values[idx].item():.3f}.png')
        elif args.vis_seg:
            pred_dicts = clip_post_classifier(data_dict, pred_dicts, keep_images=True, class_colors=PALETTE)

            print('vis seg', clip_post_classifier.vis_images.shape)
            visualiser(data_dict, pred_dicts, vis_images=clip_post_classifier.vis_images, image_name='vis_seg.png')
            return
        # elif args.vis_glip:
            # pred_dicts = clip_post_classifier(data_dict, pred_dicts, relabel=True)
            

        # visualise
        visualiser(data_dict, pred_dicts)

        if args.save_blender:
            save_blender_vis(data_dict, pred_dicts, knowns_only=args.knowns_only)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
