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
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN

from ..model_utils import model_nms_utils
import time
import cv2


class GTProposals(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True
    ):
        super(GTProposals, self).__init__()

    def forward(self, batch_dict):
        bboxes = self.get_bboxes(batch_dict)
        batch_dict['final_box_dicts'] = bboxes

        assert not self.training, "not trainable!"
        return batch_dict

    def get_bboxes(self, batch_dict):
        empty_dict = dict(pred_boxes=[], pred_scores=[], pred_labels=[])
        ret_dict = [empty_dict] * batch_dict['batch_size']

        gt_boxes_batch = batch_dict['gt_boxes']

        for k in range(batch_dict['batch_size']):
            b_gt_boxes = batch_dict['gt_boxes'][k]
            gt_labels = b_gt_boxes[..., -1].long()
            gt_boxes = b_gt_boxes[..., :7]

            is_empty = gt_labels > 10
            gt_labels = gt_labels[~is_empty]
            gt_boxes = gt_boxes[~is_empty]

            ret_dict[k]['pred_boxes'] = gt_boxes
            ret_dict[k]['pred_scores'] = torch.ones_like(gt_labels, dtype=torch.float32)
            ret_dict[k]['pred_labels'] = gt_labels

        return ret_dict 

