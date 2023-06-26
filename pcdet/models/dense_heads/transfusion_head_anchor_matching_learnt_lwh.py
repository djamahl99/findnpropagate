import copy
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
import os
torch.autograd.set_detect_anomaly(True)

def vecify_vals(anchors: torch.tensor):
    # could do bins instead :)
    bins = int(anchors.max() / anchors.min())
    place_values = [anchors.min() for i in range(bins)]

    out = []
    for anchor in anchors:
        new_vec = []
        for s in anchor:
            for pval in place_values:
                if s >= pval:
                    new_vec.append(1)
                    s -= pval
                else:
                    new_vec.append(0)
        out.append(torch.tensor(new_vec, dtype=torch.float32)[None])
    out = torch.cat(out, dim=0)

    return out

def relu_bin_vectors(anchors: torch.tensor):
    seq_len, seq_dim = anchors.shape
    num_bins = 30
    
    values = torch.linspace(anchors.min(), anchors.max(), num_bins, device=anchors.device).reshape(1, 1, -1)
    anchors = anchors.unsqueeze(2).repeat(1, 1, num_bins)

    anchors = anchors - values
    anchors = torch.relu(anchors)

    anchors = (anchors > 0) * 1.0
    anchors = anchors.reshape(seq_len, seq_dim * num_bins)

    return anchors

def hard_quantile_bin_vectors(anchors: torch.tensor, orig_anchors=None, num_bins=10):
    seq_len, seq_dim = anchors.shape

    if orig_anchors == None:
        orig_anchors = anchors.flatten()
    
    qs = torch.linspace(0.0, 1.0, num_bins, device=orig_anchors.device)
    values = torch.tensor([torch.quantile(orig_anchors, q=q) for q in qs], device=anchors.device)
    values = values.reshape(1, 1, -1)
    # values = torch.tensor([anchors.min()/2] + [torch.quantile(anchors, q=x) for x in torch.linspace(0, 1, num_bins-2)] + [anchors.max()]).reshape(1, 1, -1)
    anchors = anchors.unsqueeze(2).repeat(1, 1, num_bins)

    anchors = anchors - values
    anchors = torch.relu(anchors)
    anchors = (anchors > 0) * 1.0
    # anchors = (anchors >= values) * 1.0 + (anchors < values) * (-1.0)
    anchors = anchors.reshape(seq_len, seq_dim * num_bins)

    return anchors

def quantile_one_hot(anchors: torch.tensor, orig_anchors=None, num_bins=10):
    seq_len, seq_dim = anchors.shape

    if orig_anchors == None:
        orig_anchors = anchors.flatten()
    
    qs = torch.linspace(0, 1, num_bins, device=orig_anchors.device)
    values = torch.tensor([torch.quantile(orig_anchors, q=q) for q in qs], device=anchors.device)
    values = values.reshape(1, 1, -1)
    anchors = anchors.unsqueeze(2).repeat(1, 1, num_bins)

    anchors = anchors - values
    anchors = torch.abs(anchors)
    # anchors = (anchors > values) * 1.0
    anchors_closest_idx = torch.argmin(anchors, dim=-1)
    anchors_oh = torch.zeros_like(anchors).reshape(-1)
    anchors_oh[anchors_closest_idx] = 1.0

    return anchors_oh

def known_labels_to_full_idx(labels: torch.tensor, known_class_idx: list):
    # got_labels = set()
    for i in range(labels.shape[0]):
        if labels[i].item() < len(known_class_idx):
            # only modify if not the non-match idx
            labels[i] = known_class_idx[labels[i].item()]

            # got_labels.add(labels[i].item())

    return labels

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
    
class TransFusionHeadAnchorMatching(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHeadAnchorMatching, self).__init__()

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
        self.anchor_size_bins = 30
        self.text_dim = 3 * self.anchor_size_bins # to match what relu_bin_vectors had
        self.pred_text_dim = 3
        self.text_classes = self.anchors.shape[0]

        # self.text_features = vecify_vals(torch.tensor(anchors)) # use relu_bin_vectors in future
        # self.anchor_vecs = hard_quantile_bin_vectors(self.anchors, num_bins=self.anchor_size_bins)

        qs = torch.linspace(0.0, 1.0, self.anchor_size_bins, device=self.anchors.device)
        values = torch.tensor([torch.quantile(self.anchors, q=q) for q in qs], device=self.anchors.device)

        self.quantile_values = WrappedParameter(value=values)

        # text_features is normalized
        # self.anchor_vecs_normed = self.anchor_vecs.clone()
        # self.anchor_vecs_normed = self.anchor_vecs_normed / (1e-8 + torch.norm(self.anchor_vecs_normed, dim=1, keepdim=True))
        
        # self.sim = self.anchor_vecs_normed @ self.anchor_vecs_normed.t()
        # save_image(self.sim.unsqueeze(0).unsqueeze(0), 'sim_init.png', normalize=True)
        # print("anchor_vecs_normed", self.anchor_vecs_normed)
        # print("anchor_vecs_normed", self.anchor_vecs_normed.min(), self.anchor_vecs_normed.max())

        # self.no_gt_match_soft_value = 0.0 # should correspond to cosine sim of zero
        # # sim matrix for soft labels with extra dim of zeros
        # sim_full = torch.zeros((self.sim.shape[0] + 1, self.sim.shape[1])) + self.no_gt_match_soft_value
        # sim_full[:self.sim.shape[0], :self.sim.shape[1]] = self.sim
        # self.sim_full = sim_full
        # save_image(self.sim_full.unsqueeze(0).unsqueeze(0), 'sim_full.png', normalize=True)

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=self.pred_text_dim,kernel_size=3,padding=1))
        # layers.append(nn.BatchNorm2d(hidden_channel)) # instead of normalizing, use
        # layers.append(nn.ReLU()) # instead of normalizing, use
        # layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=self.text_dim,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)

        # self.logit_scale = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(1 / 0.07), requires_grad=True)
        # self.logit_bias = nn.Parameter(torch.ones([]), requires_grad=True)
        self.logit_scale = WrappedParameter(torch.ones(1, requires_grad=True) * np.log(1 / 0.07))
        self.logit_bias = WrappedParameter(torch.randn(1))
        self.agnostic_val = WrappedParameter(torch.randn(1))

        self.use_agnostic_heatmap = True
        if self.use_agnostic_heatmap:
            layers = []
            layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
            layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=1,kernel_size=3,padding=1))
            # # layers.append(nn.Sigmoid())
            self.agnostic_heatmap_head = nn.Sequential(*layers)

        # self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, 1)
        # self.anchor_encoding = nn.Sequential(
        #     nn.Linear(3, self.text_dim),
        #     nn.Tanh(),
        # )
        self.anchor_query_encoding = nn.Conv1d(self.text_dim, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.pred_text_dim, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, hidden_channel, 1, heads, use_bias=bias)

        self.cosine_sim = nn.CosineSimilarity(dim=1)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.forward_ret_dict = {}

    def get_lwh_vecs(self, lwh: torch.tensor):
        """Converts LWH anchors to vectors for cosine similarity

        Args:
            lwh (torch.tensor): anchors/pred size of shape (num_anchors, 3)

        Returns:
            torch.tensor: vectors of shape (num_anchors, 3*num_bins)
        """
        values = self.quantile_values()
        
        seq_len, seq_dim = lwh.shape
        num_bins = self.anchor_size_bins
        
        values = values.reshape(1, 1, -1)
        vecs = lwh.unsqueeze(2).repeat(1, 1, num_bins)

        vecs = vecs - values
        vecs = torch.relu(vecs)

        vecs = vecs / (vecs.abs() + 1e-5)
        vecs = vecs.reshape(seq_len, seq_dim * num_bins)

        return vecs

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

    def predict(self, inputs):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # get encoding for anchors
        anchor_vecs = self.get_lwh_vecs(self.anchors)
        anchor_vecs_normed = anchor_vecs.clone() / (1e-5 + torch.norm(anchor_vecs.clone(), dim=1, keepdim=True))

        self.anchor_vecs = anchor_vecs
        self.anchor_vecs_normed = anchor_vecs_normed

        if self.use_agnostic_heatmap:
            agnostic_heatmap = self.agnostic_heatmap_head(lidar_feat)
            agnostic_heatmap = self.sigmoid(agnostic_heatmap)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        N, C, H, W = dense_heatmap.shape
        dense_heatmap = rearrange(dense_heatmap, 'N C H W -> (N H W) C')
        dense_heatmap = self.get_lwh_vecs(dense_heatmap)
        dense_heatmap = rearrange(dense_heatmap, '(N H W) C -> N C H W', N=N, H=H, W=W).contiguous()
        # normalize the object size prediction
        dense_heatmap = dense_heatmap.clone() / (1e-5 + torch.norm(dense_heatmap.clone(), dim=1, keepdim=True))
        
        regression_heatmap = dense_heatmap.clone()

        N, T, H, W = dense_heatmap.shape
        dense_heatmap = rearrange(dense_heatmap, 'N T H W -> (N H W) T')

        # text_features = self.text_features.to(dense_heatmap.dtype)

        logit_scale = self.logit_scale().exp()
        dense_heatmap = logit_scale * dense_heatmap @ self.anchor_vecs_normed.t() #+ self.logit_bias()
        dense_heatmap = rearrange(dense_heatmap, '(N H W) C -> N C H W', N=N, H=H, W=W).contiguous()


        # dense_heatmap = self.heatmap_softmax(dense_heatmap)

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


        # add category embedding
        # one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        # query_cat_encoding = self.class_encoding(one_hot.float())
        # query_feat += query_cat_encoding

        # anchor vecs are similarly one-hot
        anchor_vecs = self.anchor_vecs[top_proposals_class.view(-1)]
        anchor_vecs = anchor_vecs.reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1).to(heatmap.device)

        query_anchor_encoding = self.anchor_query_encoding(anchor_vecs)
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
        sep_heatmap = self.get_lwh_vecs(sep_heatmap)
        sep_heatmap = sep_heatmap / (1e-5 + torch.norm(sep_heatmap, dim=1, keepdim=True))
        sep_heatmap_embs = sep_heatmap.clone().reshape(N, S, -1)
        # sep_heatmap = torch.cdist(sep_heatmap, text_features)
        sep_heatmap = logit_scale * sep_heatmap @ self.anchor_vecs_normed.t() + self.logit_bias()
        sep_heatmap = rearrange(sep_heatmap, '(N S) C -> N C S', N=N, S=S).contiguous()

        res_layer['heatmap'] = sep_heatmap
        res_layer['sep_heatmap_embs'] = sep_heatmap_embs
        # print('res_layer heatmap after', res_layer['heatmap'].shape)

        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

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
        # print("dense_heatmap shape before self.known_class_idx", pred_dicts["dense_heatmap"].shape)
        # print("heatmap shape before self.known_class_idx", heatmap.shape)
        pred_dicts["dense_heatmap"] = pred_dicts['dense_heatmap'][:, self.known_class_idx].contiguous()
        # pred_dicts["dense_heatmap"] = self.heatmap_softmax(pred_dicts["dense_heatmap"])
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
        # loss_reg_hmp = F.l1_loss(pred_dicts["regression_heatmap"], hm_reg)
        # loss_dict["loss_regression_heatmap"] = loss_reg_hmp.item() * self.loss_heatmap_weight * 0.1
        # loss_all += loss_reg_hmp * self.loss_heatmap_weight

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

        cls_score_sigmoid = cls_score.clone().detach().sigmoid()

        pos_labels_mask = labels < self.num_classes
        pos_labels = labels[labels < self.num_classes]
        matched_cls_score_sigmoid = cls_score_sigmoid[pos_labels_mask]
        # logging
        loss_dict[f"logit_bias"] = self.logit_bias().detach().clone()
        loss_dict[f"agnostic_val"] = self.agnostic_val().detach().clone()
        loss_dict[f"logit_scale"] = self.logit_scale().detach().clone().exp()
        loss_dict[f"non_matched_mean_conf"] = cls_score_sigmoid[labels == 10].mean()
        loss_dict[f"matched_mean_conf"] = matched_cls_score_sigmoid.mean()
        loss_dict[f"true_cls_mean_conf"] = matched_cls_score_sigmoid[F.one_hot(pos_labels, num_classes=self.num_classes).reshape(-1, 10) > 0].mean()

        assert len(self.known_class_names) == len(self.known_class_idx), "bad known idx"

        for known_idx, cls_name in zip(self.known_class_idx, self.known_class_names):
            cls_pos_labels_mask = pos_labels == known_idx
            v = matched_cls_score_sigmoid[cls_pos_labels_mask][F.one_hot(pos_labels[cls_pos_labels_mask], num_classes=self.num_classes).reshape(-1, 10) > 0]

            loss_dict[f"{cls_name}_tp_pred_conf_mean"] = v.mean()
            loss_dict[f"{cls_name}_matches"] = v.numel()

        sep_heatmap_embs = pred_dicts["sep_heatmap_embs"].permute(0, 2, 1).reshape(-1, self.text_dim)

        # log the average normed logits
        # with torch.no_grad():
        #     sep_heatmap_embs_per_dim = sep_heatmap_embs.detach().clone()
        #     pos_values = sep_heatmap_embs_per_dim[labels < self.num_classes].reshape(-1, 3, self.anchor_size_bins).permute(0, 2, 1).reshape(-1, 3)
        #     neg_values = sep_heatmap_embs_per_dim[labels == self.num_classes].reshape(-1, 3, self.anchor_size_bins).permute(0, 2, 1).reshape(-1, 3)
        #     for i, dim_name in enumerate(['length', 'width', 'height']):
        #         loss_dict[f"{dim_name}_pos_ave_sim"] = pos_values[:, i].mean()
        #         loss_dict[f"{dim_name}_neg_ave_sim"] = neg_values[:, i].mean()

        
        # regression should align with predicted class
        bbox_preds = pred_dicts['dim'].detach().clone().permute(0, 2, 1).reshape(-1, 3)
        # bbox_preds_as_targets = hard_quantile_bin_vectors(bbox_preds, orig_anchors=self.anchors, num_bins=self.anchor_size_bins)
        bbox_preds_as_targets = self.get_lwh_vecs(bbox_preds)
        bbox_alignment_loss = F.cosine_embedding_loss(sep_heatmap_embs, bbox_preds_as_targets, target=torch.ones(cls_score.shape[0], device=cls_score.device, dtype=torch.long))
        loss_dict["bbox_alignment_loss"] = bbox_alignment_loss.item()

        positive_anchors = []


        for i, lbl in enumerate(labels):
            if lbl == self.num_classes:
                # positive_anchors.append(self.anchor_vecs.mean(dim=0).reshape((1, self.text_dim)))
                # positive_anchors.append(hard_quantile_bin_vectors(self.anchors.mean(dim=0, keepdim=True), orig_anchors=self.anchors, num_bins=self.anchor_size_bins).reshape(1, self.text_dim))
                positive_anchors.append(bbox_preds_as_targets[i].reshape(1, self.text_dim))

            else:
                positive_anchors.append(self.anchor_vecs[lbl].reshape(1, self.text_dim))

        positive_anchors = torch.concat(tuple(positive_anchors), dim=0).to(sep_heatmap_embs.device)

        cos_embedding_loss = F.cosine_embedding_loss(sep_heatmap_embs, positive_anchors, target=torch.ones(cls_score.shape[0], device=cls_score.device, dtype=torch.long))
        loss_dict[f"loss_cos_emb"] = cos_embedding_loss

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
            + cos_embedding_loss * self.loss_bbox_weight  + bbox_alignment_loss * self.loss_bbox_weight 

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
