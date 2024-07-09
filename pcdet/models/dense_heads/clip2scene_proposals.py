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
from .frustum_ov3ddet import compute_bbox

CLASSES_NUSCENES = [
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

class CLIP2SceneProposer(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True
    ):
        super(CLIP2SceneProposer, self).__init__()

        self.clip2scene_preds_path = model_cfg.get('PREDS_PATH', '/media/bigdata/uqdetche/clip2scene_preds/')
        # self.lidar_name_to_preds_path = lambda name: self.clip2scene_preds_path + name + '.pth'

        

        print('clip2scene path', self.clip2scene_preds_path)
        self.cluster_method = model_cfg.CLUSTER_METHOD
        self.cluster_together = model_cfg.CLUSTER_TOGETHER
        self.min_cluster_size = model_cfg.MIN_CLUSTER_SIZE
        self.cluster_eps = model_cfg.get('EPS', 0.5)

        print('model_cfg', model_cfg.items())
        self.bg_label = 100

        self.label_map = {0: self.bg_label}
        for k, seg_label in enumerate(CLASSES_NUSCENES):
            for v, det_label in enumerate(class_names):
                if seg_label == det_label:
                    self.label_map[(k + 1)] = (v + 1) 
                    # print(f'{seg_label}[{k + 1}] => {det_label}[{v+1}]')
        
        for k, seg_label in enumerate(CLASSES_NUSCENES):
            if (k + 1) not in self.label_map:
                self.label_map[(k + 1)] = self.bg_label
                # print(f'{seg_label}[{k + 1}] => b.g.')
                

        
    def frame_to_preds_path(self, frame_id):
        # something like:
        # n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151527397766.pcd

        return self.clip2scene_preds_path + frame_id + '.bin.pth'

    def cluster_feats(self, X):
        min_cluster_size = self.min_cluster_size
        if self.min_cluster_size >= X.shape[0]:
            min_cluster_size = X.shape[0] - 1
            # no clustering as not enough points
            # return np.ones(shape=(X.shape[0], ), dtype='int')

        if min_cluster_size <= 2:
            return np.zeros(shape=(X.shape[0], ), dtype='int')

        if self.cluster_method == 'HDBSCAN':
            clustering = HDBSCAN(min_cluster_size=min_cluster_size).fit(X)
        else:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=min_cluster_size).fit(X)

        return clustering.labels_

    def get_proposals(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """

        frame_ids = batch_dict['frame_id']

        batch_size = batch_dict['batch_size']

        proposal_boxes = []
        proposal_scores = []
        frust_labels = []
        frust_batch_idx = []
        frust_scores = []

        for b in range(batch_size):
            # somehow the number of points dont matched after openpcdet preprocessing
            # cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            # get clip2scene preds
            # print('frame_id', frame_ids[b])

            lidar_path = "/media/bigdata/uqyluo/3D_DA/NUSCENES/v1.0-trainval/samples/LIDAR_TOP/" + frame_ids[b] + '.bin'
            pc_original = LidarPointCloud.from_file(lidar_path)
            points = pc_original.points.T[:, :3]
            # print('file points', points.shape)

            
            pred_path = self.frame_to_preds_path(frame_ids[b])
            preds = torch.load(pred_path, map_location='cpu')
            pred_labels = preds['predictions']
            
            # print('labels', pred_labels.shape)
            # print('cur_points', cur_points.shape)

            # map labels
            pred_labels.apply_(lambda x: self.label_map[x])
            # print('after mapping', pred_labels.min(), pred_labels.max())

            pred_labels_np = pred_labels.numpy()
            
            # mask out the background classes
            fg_mask = pred_labels != self.bg_label

            points = points[fg_mask]
            pred_labels_np = pred_labels_np[fg_mask]

            # print('X', X.shape)

            if self.cluster_together:
                X = np.concatenate((points, pred_labels_np.reshape(-1, 1)), axis=1)
                
                cluster_ids = self.cluster_feats(X)
            else:
                cluster_ids = np.zeros_like(pred_labels_np)
                cluster_id_count = 0
                
                # per class (not clustering together)
                for class_id in self.label_map.values():
                    if class_id == self.bg_label:
                        continue
                    class_mask = (pred_labels_np == class_id)

                    if class_mask.sum() == 0:
                        continue

                    X = points[class_mask]
                    print('X', X.shape)
                    clustering = DBSCAN(eps=0.25, min_samples=15).fit(X)
                    # cluster_labels = self.cluster_feats(X)
                    cluster_labels = clustering.labels_
                    
                    print('class max', class_id, cluster_labels.max(), cluster_labels.shape, [(cluster_labels==i).sum() for i in range(cluster_labels.max() + 1)])

                    cluster_ids[class_mask] = cluster_labels + cluster_id_count
                    cluster_id_count += cluster_labels.max() + 1

            for cluster_id in range(0, cluster_ids.max() + 1):
                cluster_mask = (cluster_ids == cluster_id)
                if cluster_mask.sum() == 0:
                    continue

                xyz = points[cluster_mask]
                cluster_labels = pred_labels_np[cluster_mask]
                # print('cluster_labels', cluster_labels)
                counts = np.bincount(cluster_labels)
                # print('label', np.argmax(counts))

                # x1, x2 = xyz[..., 0].min(), xyz[..., 0].max()
                # y1, y2 = xyz[..., 1].min(), xyz[..., 1].max()
                # z1, z2 = xyz[..., 2].min(), xyz[..., 2].max()

                # xc = (x1 + x2) / 2
                # yc = (y1 + y2) / 2
                # zc = (z1 + z2) / 2
                # l, w, h = (x2 - x1), (y2 - y1), (z2 - z1)

                # if l > 15 or w > 15 or h > 15:
                #     continue

                # if l < 0.3 or w < 0.3 or h < 0.3:
                #     continue

                # xyz_np = xyz.detach().cpu().numpy()

                if xyz.shape[0] < 10:
                    continue

                xc, yc, zc, l, w, h, yaw = compute_bbox(xyz)

                proposal_boxes.append(torch.tensor([xc, yc, zc, l, w, h, -1*yaw], dtype=torch.float32).reshape(1, 7))
                frust_labels.append(np.argmax(counts))
                frust_scores.append(1.0)
                frust_batch_idx.append(b)


        if len(proposal_boxes) > 0:
            proposal_boxes = torch.cat(proposal_boxes, dim=0).reshape(-1, 7).to('cuda')
            # proposal_scores = torch.tensor(proposal_scores, dtype=torch.float32)
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

