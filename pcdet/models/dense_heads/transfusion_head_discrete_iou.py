import copy
import time
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from torchvision.utils import save_image
from torchvision.ops import generalized_box_iou
from torch.distributions import Normal, kl_divergence
import os
torch.autograd.set_detect_anomaly(True)


def known_labels_to_full_idx(labels: torch.tensor, known_class_idx: list):
    # got_labels = set()
    for i in range(labels.shape[0]):
        if labels[i].item() < len(known_class_idx):
            # only modify if not the non-match idx
            labels[i] = known_class_idx[labels[i].item()]

    return labels

def box_area(boxes, min=0):
    if boxes.shape[1] == 3:
        return (boxes[:, 0] - min) * (boxes[:, 1] - min) * (boxes[:, 2] - min)

    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size//2, bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size//2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class WrappedParameter(nn.Module):
    def __init__(self, value) -> None:
        super().__init__()

        self.value = nn.Parameter(data=value, requires_grad=True)

    def forward(self):
        return self.value
    
class Threshold1d(nn.Module):
    def __init__(self, thresholds: torch.tensor) -> None:
        super().__init__()

        self.thresholds = thresholds
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.tensor):
        res = x - self.thresholds
        res = self.relu(res)

        res = res / (res.abs() + 1e-8)

        return res


class TransFusionHeadDiscreteIOU(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHeadDiscreteIOU, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = 10 # manual for one_hot

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = class_names

        self.known_class_idx = [i for i, cls in enumerate(self.all_class_names) if cls in self.known_class_names]
        print("known class idx", [(i, self.all_class_names[i]) for i in self.known_class_idx])

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        num_heads = self.model_cfg.NUM_HEADS
        dropout = self.model_cfg.DROPOUT
        activation = self.model_cfg.ACTIVATION
        ffn_channel = self.model_cfg.FFN_CHANNEL
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        self.code_size = 10

        # text_metainfo_path = 'nuscenes_text.pkl'
        # if os.path.exists(text_metainfo_path):
        #     text_metainfo = torch.load(text_metainfo_path)
        #     self.text_features = text_metainfo['text_features'].to('cuda')
        #     self.text_classes, self.text_dim = self.text_features.shape
        #     text_logit_scale = text_metainfo['logit_scale']

        #     print("Got stored text features", self.text_features.shape)
        # else:
        #     raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")

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
        self.anchors = torch.tensor(anchors, dtype=torch.float32, requires_grad=False, device='cuda')
        self.anchors = self.anchors.log()
        self.anchors_min = np.log(0.01)

        self.anchor_boxes = torch.zeros((self.anchors.shape[0], 6), device=self.anchors.device)
        self.anchor_boxes[..., 0:3] = self.anchors_min
        self.anchor_boxes[..., 3:] = self.anchors

        print("anchor boxes", self.anchor_boxes)

        self.text_dim = 3 # to match what relu_bin_vectors had
        self.text_classes = self.anchors.shape[0]

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=3,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)

        # self.logit_scale = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(1 / 0.07), requires_grad=False)
        # self.logit_bias = nn.Parameter(torch.ones([]), requires_grad=True)
        self.logit_scale = WrappedParameter(torch.ones(1, requires_grad=True) * 5)
        self.logit_bias = WrappedParameter(torch.ones(1) * -4)
        self.agnostic_val = WrappedParameter(torch.ones(1))
        # self.proposal_weight = WrappedParameter(torch.ones(1))

        self.use_agnostic_heatmap = True
        if self.use_agnostic_heatmap:
            layers = []
            layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
            layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=1,kernel_size=3,padding=1))
            # # layers.append(nn.Sigmoid())
            self.agnostic_heatmap_head = nn.Sequential(*layers)

        self.anchor_value_encoding = nn.Conv1d(3, hidden_channel, 1, bias=False)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=3, num_conv=self.model_cfg.NUM_HM_CONV)
        if self.use_agnostic_heatmap:
            heads['agnostic_head'] = dict(out_channels=1, num_conv=self.model_cfg.NUM_HM_CONV)

        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.forward_ret_dict = {}

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

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def calc_iou(self, boxes1: torch.tensor, boxes2: torch.tensor):
        lt = torch.tensor([self.anchors_min, self.anchors_min, self.anchors_min], device=boxes1.device).reshape(1, 1, 3)

        area1 = box_area(boxes1, min=self.anchors_min)
        area2 = box_area(boxes2)

        # lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
        rb = torch.min(boxes1[:, None, :3], boxes2[:, 3:])  # [N,M,3]

        lwh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]  # [N,M]

        union = area1[:, None] + area2 - inter

        return inter / (union + 1e-8)

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        if self.use_agnostic_heatmap:
            agnostic_heatmap = self.agnostic_heatmap_head(lidar_feat)
            agnostic_heatmap = self.sigmoid(agnostic_heatmap)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        regression_heatmap = dense_heatmap.clone()

        N, T, H, W = dense_heatmap.shape
        dense_heatmap = rearrange(dense_heatmap, 'N T H W -> (N H W) T')

        logit_scale = self.logit_scale().exp()
        dense_iou = self.calc_iou(dense_heatmap, self.anchor_boxes)
        dense_heatmap = logit_scale * dense_iou + self.logit_bias()
        dense_heatmap = rearrange(dense_heatmap, '(N H W) C -> N C H W', N=N, H=H, W=W).contiguous()

        # apply agnostic heatmap as element-wise weighting of heatmap
        if self.use_agnostic_heatmap:
            dense_heatmap = dense_heatmap + self.agnostic_val() * agnostic_heatmap.detach().clone()
        else:
            dense_heatmap = dense_heatmap + self.agnostic_val()
        # dense_heatmap = dense_heatmap + self.agnostic_val() * agnostic_heatmap.clone()

        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        # if self.dataset_name == "nuScenes":
        #     local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
        #     local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # # for Pedestrian & Cyclist in Waymo
        # elif self.dataset_name == "Waymo":
        #     local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
        #     local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
 
        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
            ..., : self.num_proposals
        ]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )
        self.query_labels = top_proposals_class

        anchor_vals = self.anchors[top_proposals_class.view(-1)]
        anchor_vals = anchor_vals.reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1).to(heatmap.device)

        # proposal_anchors = self.anchors[top_proposals_class.view(-1)]
        # proposal_anchors = proposal_anchors.reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1).to(heatmap.device)

        query_anchor_encoding = self.anchor_value_encoding(anchor_vals)
        query_feat += query_anchor_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        # convert to xy
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])

        query_feat = self.decoder(
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )
        res_layer = self.prediction_head(query_feat)

        # calculate classes for the separate head
        # print('res_layer heatmap', res_layer['heatmap'].shape)
        N, T, S = res_layer['heatmap'].shape
        sep_heatmap = rearrange(res_layer['heatmap'],  'N T S -> (N S) T')
        agnostic_head = rearrange(res_layer['agnostic_head'], 'N T S -> (N S) T')

        # sep_heatmap = sep_heatmap / (1e-8 + torch.norm(sep_heatmap, dim=1, keepdim=True))
        sep_heatmap_embs = sep_heatmap.clone().reshape(N, S, T)
        sep_heatmap = self.calc_iou(sep_heatmap, self.anchor_boxes)
        # print("sep iou", sep_heatmap.min(), sep_heatmap.max())
        sep_heatmap = logit_scale * sep_heatmap + self.logit_bias() + agnostic_head
        # sep_heatmap = torch.softmax(sep_heatmap, dim=1)
        sep_heatmap = rearrange(sep_heatmap, '(N S) C -> N C S', N=N, S=S).contiguous()

        res_layer['heatmap'] = sep_heatmap
        res_layer['sep_heatmap_embs'] = sep_heatmap_embs

        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        # res_layer['dim'] = res_layer["dim"] + proposal_anchors * self.proposal_weight()

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap
        res_layer['regression_heatmap'] = regression_heatmap

        if self.use_agnostic_heatmap:
            res_layer['agnostic_heatmap'] = agnostic_heatmap

        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']
        res = self.predict(feats)
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
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap
        

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

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride) 
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)


        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None])

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
    
    def heatmap_softmax(self, x):
        return torch.softmax(x, dim=1)
    
    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # remove unkown class predictions
        pred_dicts["dense_heatmap"] = pred_dicts['dense_heatmap'][:, self.known_class_idx].contiguous()
        heatmap = heatmap[:, :len(self.known_class_idx)].contiguous()

        N, C, H, W = heatmap.shape
    
        # with torch.no_grad():
        #     pred_hm = (pred_dicts["dense_heatmap"]).detach().clone().sigmoid()
        #     print("pred_hm range", pred_hm.min(), pred_hm.max())
        #     save_image(pred_hm.reshape(N*C, 1, H, W), 'pred_hm.png', nrow=C, normalize=False, scale_each=False)
        #     # save_image(torch.sigmoid(pred_dicts["dense_heatmap"].detach().clone().reshape(N*C, 1, H, W)), 'pred_hm2.png', normalize=False)
        #     save_image(heatmap.detach().clone().reshape(N*C, 1, H, W), 'true_hm.png', nrow=C, scale_each=True)

        #     if self.use_agnostic_heatmap:
        #         save_image(pred_dicts['agnostic_heatmap'].detach().clone().reshape(-1, 1, H, W), 'pred_agnostic.png', nrow=1, normalize=False)

        if self.use_agnostic_heatmap:
            loss_heatmap_func = loss_utils.FocalLossCenterNet()
            loss_agnostic_heatmap = loss_heatmap_func(pred_dicts['agnostic_heatmap'], heatmap.sum(dim=1, keepdim=True).clip(0.0, 1.0))
            loss_dict["loss_agnostic_heatmap"] = loss_agnostic_heatmap.item() * self.loss_heatmap_weight
            loss_all += loss_agnostic_heatmap * self.loss_heatmap_weight

        # regression heatmap loss
        hm_reg = heatmap.clone().permute(0, 2, 3, 1).reshape(-1, C) @ self.anchor_boxes[self.known_class_idx, 3:]
        hm_reg = hm_reg.reshape(N, H, W, 3).permute(0, 3, 1, 2)

        # save_image(hm_reg.clone().sum(dim=1), 'hm_reg_1.png')
        
        loss_reg_hmp = F.l1_loss(pred_dicts["regression_heatmap"], hm_reg)
        loss_dict["loss_regression_heatmap"] = loss_reg_hmp.item() #* self.loss_heatmap_weight * 0.1
        loss_all += loss_reg_hmp * self.loss_heatmap_weight

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)

        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        # relabel for known classes (as gaps due to removing unknowns)!
        labels = known_labels_to_full_idx(labels, self.known_class_idx)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        # assert labels[labels < self.num_classes].max() < len(self.known_class_idx), f"unknown leakage labels = {labels[labels < self.num_classes]}"
        # assert self.sim_full[-1, 0] == self.no_gt_match_soft_value, f"sim full final not correct {self.sim_full}"
        assert all([x in self.known_class_idx or x == self.num_classes for x in labels]), f"all labels should be in known idx {labels}"

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]

        # one_hot_targets = self.sim_full[labels].to(cls_score.device)

        cls_score_sigmoid = cls_score.clone().detach().sigmoid()

        pos_labels_mask = labels < self.num_classes
        pos_labels = labels[labels < self.num_classes]
        matched_cls_score_sigmoid = cls_score_sigmoid[pos_labels_mask]
        # logging
        loss_dict[f"logit_bias"] = self.logit_bias.value.detach().clone()
        loss_dict[f"agnostic_val"] = self.agnostic_val.value.detach().clone()
        loss_dict[f"logit_scale"] = self.logit_scale.value.detach().clone().exp()
        # loss_dict[f"proposal_weight"] = self.proposal_weight.value.detach().clone()
        loss_dict[f"non_matched_mean_conf"] = cls_score_sigmoid[labels == 10].mean()
        loss_dict[f"matched_mean_conf"] = matched_cls_score_sigmoid.mean()
        loss_dict[f"true_cls_mean_conf"] = matched_cls_score_sigmoid[F.one_hot(pos_labels, num_classes=self.num_classes).reshape(-1, 10) > 0].mean()

        assert len(self.known_class_names) == len(self.known_class_idx), "bad known idx"

        # bbox_targets_dim = bbox_targets.detach().clone().reshape(-1, 10)[:, 3:6].exp()
        # bbox_targets_dim = bbox_targets_dim[pos_labels_mask]

        for known_idx, cls_name in zip(self.known_class_idx, self.known_class_names):
            cls_pos_labels_mask = pos_labels == known_idx
            v = matched_cls_score_sigmoid[cls_pos_labels_mask][F.one_hot(pos_labels[cls_pos_labels_mask], num_classes=self.num_classes).reshape(-1, 10) > 0]

            loss_dict[f"{cls_name}_tp_pred_conf_mean"] = v.mean()
            loss_dict[f"{cls_name}_matches"] = v.numel()

        sep_heatmap_embs = pred_dicts["sep_heatmap_embs"].permute(0, 2, 1).reshape(-1, 3)
        
        # regression should align with predicted class
        bbox_preds = pred_dicts['dim'].detach().clone().permute(0, 2, 1).reshape(-1, 3)

        # bbox_preds_as_targets = self.soft_relu_bin_vectors(bbox_preds)
        # bbox_preds_as_targets = torch.zeros((bbox_preds.shape[0], 6), device=bbox_preds.device)
        # bbox_preds_as_targets[..., 0:3] = self.anchors_min
        # bbox_preds_as_targets[..., 3:] = bbox_preds

        bbox_alignment_loss = F.l1_loss(sep_heatmap_embs[labels < self.num_classes], bbox_preds[labels < self.num_classes])
        loss_dict["bbox_alignment_loss"] = bbox_alignment_loss.item()

        positive_anchors = self.anchor_boxes[pos_labels, 3:]

        box_reg_loss = F.l1_loss(sep_heatmap_embs[pos_labels_mask], positive_anchors)
        loss_dict[f"box_reg_loss"] = box_reg_loss

        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight \
            + box_reg_loss * 0.1  + bbox_alignment_loss * 0.1 

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        code_size = 10
        targets = torch.zeros([bboxes.shape[0], code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
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
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

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

        batch_size = preds_dicts["heatmap"].shape[0]
        batch_score = preds_dicts["heatmap"].sigmoid()
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dicts["query_heatmap_score"] * one_hot
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
            filter=True,
        )
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
