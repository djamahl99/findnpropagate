import copy
from typing import List, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from pcdet.utils.box_utils import boxes_to_corners_3d
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils

from ..model_utils import model_nms_utils

from ..preprocessed_detector import PreprocessedDetector
from pcdet.utils import common_utils

from ..backbones_3d import PointNet2MSG

from pcdet.models.frustum_pointnets_v1 import FrustumPointNetv1

class FrustumPointNetHead(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(FrustumPointNetHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.num_classes = 10 # manual for one_hot
        self.num_known_classes = len(class_names)

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = class_names

        # for k, v in self.coco_to_nuscenes_idx.items():
            # print(f'match {coco_classes[k]} with {self.all_class_names[v]}')

        self.known_class_idx = [i for i, cls in enumerate(self.all_class_names) if cls in self.known_class_names]
        print("known class idx", [(i, self.all_class_names[i]) for i in self.known_class_idx])

        self.model_cfg = model_cfg
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        self.label_to_hierarchy_label = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5}

        self.image_size = self.model_cfg.IMAGE_SIZE

        self.code_size = 10

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

        # preds_path = "/home/uqdetche/GLIP/OWL_"
        preds_path = self.model_cfg.PREDS_PATH
        camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.image_detector = PreprocessedDetector([preds_path + f"{cam_name}.json" for cam_name in camera_names])

        self.hierarchy_anchors = torch.tensor(model_cfg.get('HIERARCHY_ANCHORS'))
        print('hierarchy_anchors', self.hierarchy_anchors.shape)
        self.frustum_model = FrustumPointNetv1(n_classes=self.hierarchy_anchors.shape[0], hierarchy_anchors=self.hierarchy_anchors)
        sd = torch.load(model_cfg.CKPT, map_location='cpu')
        self.frustum_model.load_state_dict(sd)
        self.frustum_model.eval()

        self.forward_ret_dict = {}

    def to_batch_dict(self, batched_frustum_points, batched_one_hot, batch_frust_rot):
        batched_frustum_points = batched_frustum_points.permute(0, 2, 1)
        batch_dict = dict(points=batched_frustum_points, one_hot=batched_one_hot, prerot=batch_frust_rot)
        return batch_dict

    def project_to_image(self, batch_dict, batch_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        batch_size = batch_dict['batch_size']

        points = batch_dict['points']
        points_idx = points[..., 0]

        batch_mask = (points_idx == batch_idx)
        points = points[batch_mask, 1:4]

        cur_coords = points.clone()

        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        cur_img_aug_matrix = img_aug_matrix[batch_idx]
        cur_lidar_aug_matrix = lidar_aug_matrix[batch_idx]
        cur_lidar2image = lidar2image[batch_idx]

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

    def predict(self, batch_dict):
        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        # lidar2image = batch_dict['lidar2image']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        batch_size = batch_dict['batch_size']
        
        points_batch_idx = batch_dict['points'][..., 0]

        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        # total boxes in this batch
        total_boxes = det_boxes.shape[0]
        batch_box_idx = 0
        print('det_boxes', det_boxes.shape)

        self.max_points = 1024 # probably have less points than this
        self.min_points = 5 # min points to make a proposal

        # mask of which pts are in each 2D box frustum
        batch_frust_pts = torch.zeros((total_boxes, self.max_points, 3), device='cuda')
        batch_frust_rot = torch.zeros((total_boxes), device='cuda')
        # from 2D detector
        batch_scores = torch.zeros((total_boxes), device='cuda')
        batch_labels = torch.zeros((total_boxes), dtype=torch.long, device='cuda')

        # keep track of batch indx
        batch_idx = torch.zeros((total_boxes), dtype=torch.long, device='cuda')

        # lidar pts
        points = batch_dict['points']
        points_idx = points[..., 0] 

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            # sort 2D boxes by highest score (as we are limiting to 200 proposals per batch like transfusion)
            # indices = torch.sort(cur_scores, descending=True).indices
            # cur_boxes, cur_labels, cur_scores, cur_cam_idx = cur_boxes[indices], cur_labels[indices], cur_scores[indices], cur_cam_idx[indices]

            batch_pts_mask = (points_idx == b)

            proj_points, proj_points_cam_mask = self.project_to_image(batch_dict, batch_idx=b)

            for c in range(6): # 6 cameras
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                cam_points = proj_points[c, proj_points_cam_mask[c]]
                assert cam_points.numel() > 0, "no points on this view!"

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < 0.1:
                        continue

                    x1, y1, x2, y2 = box.cpu()
                    in_cam_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    # box_points = cam_points[in_cam_box]
                    comb_mask = proj_points_cam_mask[c].clone().reshape(-1)
                    comb_mask[proj_points_cam_mask[c].reshape(-1)] = in_cam_box

                    box_lidar_points = cur_points[comb_mask]

                    if box_lidar_points.shape[0] < self.min_points: # filter boxes with little points
                        continue


                    center = box_lidar_points.mean(dim=0)
                    print('center', center)
                    frust_rot = torch.atan2(center[1], center[0])

                    print('box_lidar pre', box_lidar_points.shape)
                    box_lidar_points = common_utils.rotate_points_along_z(box_lidar_points[None], -frust_rot.reshape(-1))[0]
                    print('box_lidar after', box_lidar_points.shape)

                    pt_indices = torch.randint(0, box_lidar_points.shape[0], (self.max_points,))
                    batch_frust_pts[batch_box_idx, :self.max_points] = box_lidar_points[pt_indices]
                    batch_frust_rot[batch_box_idx] = frust_rot


                    # info from the 2D detector
                    batch_labels[batch_box_idx] = self.label_to_hierarchy_label[int(label)]
                    batch_scores[batch_box_idx] = score

                    batch_idx[batch_box_idx] = b

                    batch_box_idx += 1

        self.query_labels = batch_labels.unsqueeze(0)
        # class output (purely based on the 2D detector)
        hierarchy_labels = set(self.label_to_hierarchy_label)
        one_hot = F.one_hot(batch_labels, num_classes=self.hierarchy_anchors.shape[0])

        print('one_hot', one_hot.shape)
        print('points', batch_frust_pts.shape)

        frust_batch = self.to_batch_dict(batch_frust_pts, one_hot, batch_frust_rot)

        with torch.no_grad():
            frust_preds = self.frustum_model(frust_batch, pred=True)

        # undo the batch dimension
        num_in_batch = {b: 0 for b in range(batch_size)}
        for b in range(batch_size):
            num_in_batch[b] = int((batch_idx == b).sum())

        print('num_in_batch', num_in_batch.items())
        # batched_predictions = 

        assert batch_size == 1, 'not implemented batchsize > 1'

        res_layer = dict(heatmap=one_hot.unsqueeze(0))
        for k in frust_preds.keys():
            res_layer[k] = frust_preds[k].unsqueeze(0)
            print(f'res_layer[{k}] => {res_layer[k].shape}')
        # add batch back
    

        return res_layer

    def forward(self, batch_dict):
        res = self.predict(batch_dict)
        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[...,:-1]
            gt_labels_3d =  gt_boxes[...,-1].long() - 1
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx : batch_idx + 1]
            gt_bboxes = gt_bboxes_3d[batch_idx]
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = torch.cat(res_tuple[5], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious
        

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        
        num_proposals = preds_dict["center"].shape[-1]
        score = copy.deepcopy(preds_dict["heatmap"].detach())
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)
        bboxes_tensor = boxes_dict[0]["pred_boxes"]
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 9)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), ious[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        matched_ious = matched_ious.reshape(-1)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        pos_labels_mask = labels < self.num_classes
        pos_labels = labels[labels < self.num_classes]
        matched_cls_score_sigmoid = cls_score[pos_labels_mask]

        for known_idx, cls_name in zip(self.known_class_idx, self.known_class_names):
            cls_pos_labels_mask = pos_labels == known_idx
            v = matched_cls_score_sigmoid[cls_pos_labels_mask][F.one_hot(pos_labels[cls_pos_labels_mask], num_classes=self.num_classes).reshape(-1, 10) > 0]
            v_ious = matched_ious[labels == known_idx]

            num_matches = v.numel() if v.numel() is not None else 0
            loss_dict[f"{cls_name}_tp_pred_conf_mean"] = v.mean()
            loss_dict[f"{cls_name}_matches"] = num_matches
            loss_dict[f"{cls_name}_iou_mean"] = v_ious.mean()

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        # stage1_center = pred_dicts["stage1_center"].permute(0, 2, 1)
        # loss_stage1 = self.loss_bbox(stage1_center, bbox_targets[:, :, :3]).sum() / max(num_pos, 1)

        # loss_dict["loss_stage1_center"] = loss_stage1.item()

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_bbox * self.loss_bbox_weight + loss_cls * self.loss_cls_weight

        loss_dict[f"matched_ious"] = matched_ious[labels < self.num_classes].mean()
        loss_dict['loss_trans'] = loss_all #+ loss_stage1 * self.loss_bbox_weight

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        # targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        # targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])

        # targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
        # targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.point_cloud_range[4] - self.point_cloud_range[1])

        targets[:, 0] = bboxes[:, 0]
        targets[:, 1] = bboxes[:, 1]

        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(-1, keepdims=False).indices
        final_scores = heatmap.max(-1, keepdims=False).values

        # center[:, 0, :] = center[:, 0, :] * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
        # center[:, 1, :] = center[:, 1, :] * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
        # dim = dim.exp()
        # rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        # rot = torch.atan2(rots, rotc)
        # rot

        vs = [center, height, dim, rot]
        names = 'center, height, dim, rot, vel'.split(', ')

        print('decode_box')
        for v, name in zip(vs, names):
            print(f'{name} => {v.shape}')

        print('heatmap', heatmap.shape)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=-1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=-1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh        
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"]
        # one_hot = F.one_hot(
        #     self.query_labels, num_classes=self.num_classes
        # ).permute(0, 2, 1)
        # batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=False,
        )
        for k in range(batch_size):
            # print(post_process_cfg)
            # selected, selected_scores = model_nms_utils.class_agnostic_nms(
            #     box_scores=ret_dict[k]['pred_scores'], box_preds=ret_dict[k]['pred_boxes'],
            #     nms_config=post_process_cfg.NMS_CONFIG,
            #     score_thresh=None
            # )
            # print('selected', selected.shape, ret_dict[k]['pred_boxes'].shape)
            # ret_dict[k]['pred_boxes'] = ret_dict[k]['pred_boxes'][selected]
            # ret_dict[k]['pred_scores'] = selected_scores
            # ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'][selected]
            print('preds', ret_dict[k]['pred_boxes'].shape)

            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
