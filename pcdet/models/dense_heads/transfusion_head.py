import copy
from matplotlib import pyplot as plt
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
from pcdet.models.dense_heads.pseudo_processor import PseudoProcessor

from torchvision.utils import save_image
from pcdet.ops.iou3d_nms import iou3d_nms_utils


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



class TransFusionHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHead, self).__init__()

        self.use_pseudo = model_cfg.get('USE_PSEUDO', False)
        self.balanced_reweighting = model_cfg.get('BALANCED_REWEIGHTING', False)
        self.save_preds = model_cfg.get('SAVE_PREDS', False)
        num_class = model_cfg.get('NUM_CLASSES', num_class)
        
        self.relabel_classes = 'KNOWN_CLASS_NAMES' in model_cfg
        if self.relabel_classes:
            assert 'KNOWN_CLASS_NAMES' in model_cfg and 'FULL_CLASS_NAMES' in model_cfg, 'need both known and all class names given'
            self.known_class_names = model_cfg.get('KNOWN_CLASS_NAMES')
            self.all_class_names = model_cfg.get('FULL_CLASS_NAMES')
            self.relabel_map = {(i + 1): (j + 1) for i, k_name in enumerate(self.known_class_names) 
                                for j, a_name in enumerate(self.all_class_names) if k_name == a_name}

        self.class_names = class_names
        # TODO
        # class_names = ['car', 'construction_vehicle', 'trailer', 'barrier', 'bicycle', 'pedestrian']

        if self.use_pseudo:
            self.pseudo_processor = PseudoProcessor(class_names, self_training_folder=model_cfg.SELF_TRAIN_PATH)
            num_class = self.pseudo_processor.num_classes # uses all classes to calculate
        self.pseudo_nms_thresh = model_cfg.get('PSEUDO_NMS_THRESH', None)

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_classes = num_class

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
        self.label_smoothing = self.model_cfg.LOSS_CONFIG.get('LABEL_SMOOTHING', False)
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1

        if loss_cls.get("use_poly1", False):
            self.loss_cls = loss_utils.Poly1SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha, epsilon=loss_cls.epsilon)
        elif loss_cls.get("use_mae", False):
            self.loss_cls = loss_utils.WeightedSigmoidL1Loss()
        elif loss_cls.get('use_ce', False):
            self.loss_cls = loss_utils.SigmoidClassificationLoss()
        else:
            self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']

        self.code_size = len(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride

        print('x_size', x_size, y_size)
        
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

    def predict(self, inputs, batch_dict):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        N, C, H, W = heatmap.shape
        # save_image(heatmap.clone().reshape(N*C, 1, H, W), 'pred_hm_raw.png', nrow=1)

        # heatmap_np = heatmap.detach().clone().reshape(N*C, 1, H, W).cpu().numpy()
        # C1 = int(np.ceil(np.sqrt(C)))
        # C2 = int(np.ceil(C / C1))
        # fig, axs = plt.subplots(C1, C2, figsize=(int(4*C1), int(4*C2)))
        # hm_min, hm_max = heatmap_np.min(), heatmap_np.max()

        # # print('C1', C1, C2, C, C1*C2)

        # axs = [axs[i, j] for i in range(C1) for j in range(C2)]


        # assert N == 1, 'only support debug witj batch_size==1'
        # gt_boxes = batch_dict['gt_boxes'].detach().cpu()
        # gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1])

        # gt_labels = gt_boxes[:, -1]

        # print('gt_labels', gt_labels.min(), gt_labels.max())

        # # print('gt_labels', gt_labels.shape)
        # # print('gt_boxes', gt_boxes.shape)

        # # print('number pseudo', gt_boxes.shape[0])

        # for idx, name in enumerate(self.pseudo_processor.all_class_names):
        #     gt_mask = (gt_labels == (idx + 1))
            
        #     boxes = gt_boxes[gt_mask]
        #     # print('boxes', boxes)


        #     chm = heatmap_np[idx, 0]
        #     # chm = (chm - chm.min()) / (chm.max() - chm.min())
        #     chm = (chm - hm_min) / (hm_max - hm_min)
        #     axs[idx].imshow(chm, cmap='viridis')
        #     axs[idx].set_title(name)

        #     if boxes.numel() > 0:
        #         # plot the box centres
        #         # print('boxes[:, 0]', boxes[:, 0].shape, 'boxes[: 1]', boxes[:, 1].shape)

        #         boxes[:, 0] = (boxes[:, 0] - self.point_cloud_range[0]) / (self.point_cloud_range[3] - self.point_cloud_range[0])
        #         boxes[:, 1] = (boxes[:, 1] - self.point_cloud_range[1]) / (self.point_cloud_range[4] - self.point_cloud_range[1])
                
        #         boxes[:, 0] *= W
        #         boxes[:, 1] *= H

        #         axs[idx].scatter(boxes[:, 0], boxes[:, 1], color='red')

        # plt.savefig('pred_hm.png', bbox_inches='tight')
        # plt.close()


        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.dataset_name == "nuScenes" and self.num_classes == 10:
            local_max[ :, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[ :, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        # for Pedestrian & Cyclist in Waymo
        elif self.dataset_name == "Waymo":
            local_max[ :, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        # for Pedestrian, person_sitting and cyclist in kitti
        elif self.dataset_name == "kitti":
            local_max_indices = [i for i, name in enumerate(self.class_names) if name in ['Pedestrian', 'Person_Sitting', 'Cyclist']]
            # print('local_max_indices', local_max_indices)
            # if heatmap.shape[1] == 7:
                # local_max[ :, [4, 5, 6], ] = F.max_pool2d(heatmap[:, [4, 5, 6]], kernel_size=1, stride=1, padding=0)

            if len(local_max_indices) > 0:
                local_max[ :, local_max_indices, ] = F.max_pool2d(heatmap[:, local_max_indices], kernel_size=1, stride=1, padding=0)
            
            # local_max[ :, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)

        # save_image(local_max.clone().reshape(N*C, 1, H, W), 'kitti_heatmap_localmax.png')

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
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding

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
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap

        return res_layer

    def forward(self, batch_dict):
        if self.use_pseudo and self.training:
            batch_dict = self.pseudo_processor(batch_dict)
            # print('pseudo_boxes after processing', batch_dict['pseudo_boxes'].shape)

        feats = batch_dict['spatial_features_2d']
        res = self.predict(feats, batch_dict)
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

            # if self.use_pseudo and self.pseudo_processor.self_training:
                # self.pseudo_processor.save_predictions(batch_dict, self.get_bboxes(res))
            
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
        # matched_ious = np.mean(res_tuple[5])
        matched_ious = torch.cat(res_tuple[5], dim=0)
        heatmap = torch.cat(res_tuple[6], dim=0)
        unknown_mask = torch.cat(res_tuple[7], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap, unknown_mask
        

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
        unknown_mask = torch.zeros([num_proposals], dtype=torch.bool).to(center.device)

        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.float32)

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

            for pos_ind, assigned_ind in zip(pos_inds, pos_assigned_gt_inds):
                label = gt_labels_3d[assigned_ind]

                if self.use_pseudo:
                    label1 = int(label + 1) # has to be +1 
                    assert label1 > 0, 'label should start from 1'
                    if label1 in self.pseudo_processor.unknown_labels:
                        unknown_mask[pos_ind] = True

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

                if self.use_pseudo and int(gt_labels_3d[idx] + 1) in self.pseudo_processor.unknown_labels: # unknowns have larger localisation uncertainty
                    radius = int(radius * target_assigner_cfg.UNK_RADIUS_MULT)

                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)


        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), ious[None], heatmap[None], unknown_mask[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap, unknown_mask = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # with torch.no_grad():
        #     N, C, H, W = heatmap.shape

        #     pred_hm = (pred_dicts["dense_heatmap"]).detach().clone().sigmoid()
        #     # print("pred_hm range", pred_hm.min(), pred_hm.max())
        #     save_image(pred_hm.reshape(N*C, 1, H, W), 'pred_hm.png', nrow=C, normalize=True, scale_each=True)
        #     # save_image(torch.sigmoid(pred_dicts["dense_heatmap"].detach().clone().reshape(N*C, 1, H, W)), 'pred_hm2.png', normalize=False)
        #     save_image(heatmap.detach().clone().reshape(N*C, 1, H, W), 'true_hm.png', nrow=C, normalize=True, scale_each=True)

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        matched_ious = matched_ious.reshape(-1)

        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        # Logging #################################

        cls_score_sigmoid = cls_score.detach().clone().sigmoid()
        pos_labels_mask = labels < self.num_classes
        pos_labels = labels[labels < self.num_classes]
        matched_cls_score_sigmoid = cls_score_sigmoid[pos_labels_mask]

        num_per_label = {l: 0 for l in range(self.num_classes)}

        # print('pseudo_processor', self.pseudo_processor.forward_pseudo_stats.keys())

        if self.use_pseudo:
            for idx, cls_name in enumerate(self.pseudo_processor.all_class_names):
                cls_pos_labels_mask = (pos_labels == idx)
                v = matched_cls_score_sigmoid[cls_pos_labels_mask][F.one_hot(pos_labels[cls_pos_labels_mask], num_classes=self.num_classes).reshape(-1, self.num_classes) > 0]
                v_ious = matched_ious[labels == idx]
                num_matches = v.numel() if v.numel() is not None else 0

                loss_dict[f"{cls_name}_tp_pred_conf_mean"] = v.mean()
                loss_dict[f"{cls_name}_matches"] = num_matches
                loss_dict[f"{cls_name}_iou_mean"] = v_ious.mean()
                num_per_label[idx] = self.pseudo_processor.forward_pseudo_stats[f'num_per_class_{cls_name}']

            # pseudo infos
            for k, v in self.pseudo_processor.forward_pseudo_stats.items():
                loss_dict[k] = v
        else:
            for idx, cls_name in enumerate(self.class_names):
                cls_pos_labels_mask = (pos_labels == idx)
                v = matched_cls_score_sigmoid[cls_pos_labels_mask][F.one_hot(pos_labels[cls_pos_labels_mask], num_classes=self.num_classes).reshape(-1, self.num_classes) > 0]
                v_ious = matched_ious[labels == idx]
                num_matches = v.numel() if v.numel() is not None else 0

                loss_dict[f"{cls_name}_tp_pred_conf_mean"] = v.mean()
                loss_dict[f"{cls_name}_matches"] = num_matches
                loss_dict[f"{cls_name}_iou_mean"] = v_ious.mean()

        ###########################################

        if self.balanced_reweighting:
            for lbl, cls_name in enumerate(self.pseudo_processor.all_class_names):
                lbl_weight = 1.0 / torch.clamp((labels == lbl).sum(), min=1)
                label_weights[labels == lbl] = lbl_weight

            label_weights = label_weights.numel() * (label_weights / label_weights.sum())

            # log weight
            for lbl, cls_name in enumerate(self.pseudo_processor.all_class_names):
                if (labels == lbl).sum() > 0:
                    lbl_weight = label_weights[labels == lbl].mean()
                else:
                    lbl_weight = 1.0
                
                loss_dict[f"{cls_name}_lbl_w"] = lbl_weight

        # unknown class weight
        if self.use_pseudo and 'unknown_cls_weight' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS:
            label_weights[unknown_mask.reshape(-1)] *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['unknown_cls_weight']

        if self.label_smoothing:
            # do label smoothing
            cls_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
            cls_score.fill_(self.label_smoothing / (self.num_classes - 1))
            cls_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), (1.0 - self.label_smoothing))
            cls_targets = cls_targets[..., :-1]
        else:
            # use one hot 
            cls_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
            cls_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
            cls_targets = cls_targets[..., :-1]

        loss_cls = self.loss_cls(
            cls_score, cls_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER], dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']

        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        # unknowns penalized differently for e.g. velocity and rotation
        if 'unknown_code_weights' in self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS:
            unknown_code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['unknown_code_weights']
            unknown_code_weights = bbox_weights.new_tensor(unknown_code_weights)
            reg_weights[unknown_mask] *= unknown_code_weights

        loss_bbox = self.loss_bbox(preds, bbox_targets) 
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = matched_ious[labels < self.num_classes].mean()
        loss_dict['loss_trans'] = loss_all

        return loss_all,loss_dict

    def encode_bbox(self, bboxes):
        targets = torch.zeros([bboxes.shape[0], self.code_size]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = bboxes[:, 7:]
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):
        
        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH
        score_thresh_unk = post_process_cfg.get('SCORE_THRESH_UNK', None)
        post_center_range = post_process_cfg.POST_CENTER_RANGE
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        if filter:
            is_unknown = torch.zeros_like(final_preds, dtype=torch.float)
            for b, batch_preds in enumerate(final_preds):
                for idx, label in enumerate(batch_preds):
                    label = label.item() + 1

                    if hasattr(self, 'pseudo_processor'):
                        is_unknown[b, idx] = float(label in self.pseudo_processor.unknown_labels)

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

        if score_thresh_unk is not None:
            score_threshs = score_thresh * (1 - is_unknown) + is_unknown * score_thresh_unk
        else:
            score_threshs = score_thresh

        thresh_mask = final_scores > score_threshs  
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]

            if boxes3d.shape[0] > 0 and scores.shape[0] > 0 and self.training and self.pseudo_nms_thresh: # TODO: change (necessary for the pseudo self-training)
                nms_indices, _ = iou3d_nms_utils.nms_normal_gpu(boxes3d[:, :7],
                    scores, thresh=self.pseudo_nms_thresh)

                boxes3d = boxes3d[nms_indices]
                scores = scores[nms_indices]
                labels = labels[nms_indices]

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

            
            # RELABELING: for when trained on few class, and test on all
            if self.relabel_classes:
                for i in range(ret_dict[k]['pred_labels'].numel()):
                    ret_dict[k]['pred_labels'][i] = self.relabel_map[ret_dict[k]['pred_labels'][i].item()]

        return ret_dict 
