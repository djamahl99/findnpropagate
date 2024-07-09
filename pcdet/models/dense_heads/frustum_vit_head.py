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

from .vit_point_encoder import ObjectPointsEncoder

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

class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=2):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512+n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

        self.out_features = 256

    def forward(self, pts): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        # expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        # expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        expand_global_feat = global_feat

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        # box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4

        # return box_pred

        # just want features
        return x

class STNxyz(nn.Module):
    def __init__(self,n_classes=0):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
        
    def forward(self, pts):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256


        # expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        # x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        
        return x



class FrustumViTHead(nn.Module):
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(FrustumViTHead, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.num_classes = 10 # manual for one_hot

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = class_names

        # for k, v in self.coco_to_nuscenes_idx.items():
            # print(f'match {coco_classes[k]} with {self.all_class_names[v]}')

        self.known_class_idx = [i for i, cls in enumerate(self.all_class_names) if cls in self.known_class_names]
        print("known class idx", [(i, self.all_class_names[i]) for i in self.known_class_idx])

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        self.image_size = self.model_cfg.IMAGE_SIZE

        # hidden_channel=self.model_cfg.HIDDEN_CHANNEL
        self.num_proposals = self.model_cfg.NUM_PROPOSALS
        self.bn_momentum = self.model_cfg.BN_MOMENTUM
        # self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE

        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma,alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']

        self.code_size = 10

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')


        hidden_channel = 64
        self.dim = hidden_channel
        # self.STN = STNxyz(n_classes=0)
        # self.pointnet = PointNetEstimation(n_classes=0)
        self.encoder = ObjectPointsEncoder(input_dim=3, dim=hidden_channel, depth=2, heads=8)
        # self.pointnet_out_channels = self.pointnet.out_features
        # hidden_channel = self.pointnet_out_channels
        # self.camera_embedding = nn.Embedding(num_embeddings=6, embedding_dim=hidden_channel)

        # preds_path = "/home/uqdetche/GLIP/OWL_"
        preds_path = self.model_cfg.PREDS_PATH
        camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.image_detector = PreprocessedDetector([preds_path + f"{cam_name}.json" for cam_name in camera_names])

        # a shared convolution
        # self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        # layers = []
        # layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        # layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        # self.heatmap_head = nn.Sequential(*layers)
        # self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        # self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
        #         self_posembed=PositionEmbeddingLearned(2, hidden_channel),
        #         cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
        #     )
        self.class_emb = nn.Embedding(10, hidden_channel)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_channel, nhead=8, batch_first=True, dim_feedforward=hidden_channel)

        # self.encoder_layer = nn.TransformerEncoder(self.encoder_layer, 4, )

        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        # self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)


        self.init_bn_momentum()

        self.forward_ret_dict = {}

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

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

        self.max_points = 256 # probably have less points than this
        self.min_points = 5 # min points to make a proposal

        # mask of which pts are in each 2D box frustum
        # query_box_pt_mask = torch.zeros((batch_size, self.num_proposals, points_batch_idx.shape[0]), dtype=torch.bool, device='cuda')
        query_box_pts = torch.zeros((batch_size, self.num_proposals, self.max_points, 3), device='cuda')
        # query_pts_emb = torch.zeros((batch_size, self.num_proposals, self.dim), device='cuda')

        query_pos_batch = torch.zeros((batch_size, self.num_proposals, 3), device='cuda')

        # from 2D detector
        query_score_batch = torch.zeros((batch_size, self.num_proposals, 1), device='cuda')
        query_labels_batch = torch.zeros((batch_size, self.num_proposals), dtype=torch.long, device='cuda')

        # for cam to lidar
        query_cam_idx = torch.zeros((batch_size, self.num_proposals), dtype=torch.long, device='cuda')
        query_batch_idx = torch.zeros((batch_size, self.num_proposals), dtype=torch.long, device='cuda')

        # lidar pts
        points = batch_dict['points']
        points_idx = points[..., 0] 

        for b in range(batch_size):
            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            if cur_boxes.shape[0] >= self.num_proposals:
                print('there are more detected boxes than max proposals!', self.num_proposals)

            # sort 2D boxes by highest score (as we are limiting to 200 proposals per batch like transfusion)
            # indices = torch.sort(cur_scores, descending=True).indices
            # cur_boxes, cur_labels, cur_scores, cur_cam_idx = cur_boxes[indices], cur_labels[indices], cur_scores[indices], cur_cam_idx[indices]

            batch_pts_mask = (points_idx == b)

            proj_points, proj_points_cam_mask = self.project_to_image(batch_dict, batch_idx=b)

            batch_box_idx = 0

            for c in range(6): # 6 cameras
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                cam_points = proj_points[c, proj_points_cam_mask[c]]
                assert cam_points.numel() > 0, "no points on this view!"

                if batch_box_idx > self.num_proposals:
                    break

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < 0.1:
                        continue

                    if batch_box_idx >= self.num_proposals:
                        print('too many 2d detections!', batch_box_idx, self.num_proposals, cur_boxes.shape)
                        break

                    x1, y1, x2, y2 = box.cpu()
                    in_cam_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    box_points = cam_points[in_cam_box]

                    if box_points.shape[0] < self.min_points: # filter boxes with little points
                        continue

                    # check that the 2D label is valid
                    assert label >= 0 and label < self.num_classes, f'labels should be within number of classes! {label}'

                    box_lidar_points = self.get_geometry_at_image_coords(box_points, [c] * box_points.shape[0], [b] * box_points.shape[0],
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    # box_lidar_points = (box_lidar_points - self.point_cloud_min) / (self.point_cloud_max - self.point_cloud_min)
                    
                    # curr_box_pt_mask = proj_points_cam_mask[c].clone()

                    # mask for on this camera and in this box
                    # curr_box_pt_mask[curr_box_pt_mask] = in_cam_box 
                    # query_box_pt_mask[b, batch_box_idx] = curr_box_pt_mask

                    median_bx_pts = torch.median(box_lidar_points, dim=0).values
                    box_lidar_points = box_lidar_points - median_bx_pts

                    # print('box points', box_points.shape)
                    num_pts = min(self.max_points, box_lidar_points.shape[0])
                    pt_indices = torch.linspace(0, box_lidar_points.shape[0] - 1, steps=num_pts, dtype=torch.long, device=box_lidar_points.device)
                    query_box_pts[b, batch_box_idx, :num_pts] = box_lidar_points[pt_indices]
                    # enc = self.encoder(box_lidar_points[pt_indices].reshape(1, -1, 3))
                    # print('enc', enc.shape)
                    # query_pts_emb[b, batch_box_idx, :] = enc

                    query_pos_batch[b, batch_box_idx] = median_bx_pts

                    # info from the 2D detector
                    query_labels_batch[b, batch_box_idx] = label
                    query_score_batch[b, batch_box_idx] = score
                    
                    # camera and batch indices
                    query_batch_idx[b, batch_box_idx] = b
                    query_cam_idx[b, batch_box_idx] = c

                    batch_box_idx += 1


        # box_pts_shape = query_box_pts.shape
        # query_box_pts = query_box_pts.reshape(batch_size * self.num_proposals, *box_pts_shape[2:])


        # query_box_pts = query_box_pts.permute(0, 2, 1)
        query_pos_batch = query_pos_batch.permute(0, 2, 1)

        # predict further centre delta
        # center_delta = self.STN(query_box_pts).reshape(batch_size, self.num_proposals, 3).permute(0, 2, 1)

        # stage1 center: mean + centre_delta
        stage1_center = query_pos_batch
        # query_box_pts = query_box_pts - center_delta.permute(0, 2, 1).reshape(-1, 3).unsqueeze(-1)

        # query_feat = self.pointnet(query_box_pts).reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1)

        query_feat = self.encoder(query_box_pts.reshape(batch_size * self.num_proposals, -1, 3))
        # print('query feat', query_feat.shape)
        query_feat = query_feat.reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1)
        # print('encoder feat', query_feat.shape)
        # query_feat = query_pts_emb.permute(0, 2, 1) 

        # query_feat += self.camera_embedding(query_cam_idx).permute(0, 2, 1)
        # query_feat = query_feat + self.class_emb(query_labels_batch).permute(0, 2, 1)
        # print('labels_feat', labels_feat.shape, query_feat.shape)

        # transformer layer (idea: multihead attn can attend to overlapping boxes)
        query_feat = self.encoder_layer(query_feat.permute(0, 2, 1)).permute(0, 2, 1)
    
        # print('')
        # 3D box prediction
        res_layer = self.prediction_head(query_feat)

        # camera 3D center location
        center_coords = res_layer['center'] + stage1_center
        # center_coords = center_coords.permute(0, 2, 1)

        # camera centres to lidar centres
        # center_lidar = self.get_geometry_at_image_coords(center_xyz_cam.reshape(-1, 3), query_cam_idx.reshape(-1), query_batch_idx.reshape(-1),
        #     camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
        #     post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        # ).reshape(batch_size, self.num_proposals, -1).permute(0, 2, 1)

        self.query_labels = query_labels_batch
        # class output (purely based on the 2D detector)
        one_hot = F.one_hot(query_labels_batch, num_classes=self.num_classes).permute(0, 2, 1)

        # zero for no object
        one_hot = one_hot * query_score_batch.permute(0, 2, 1)
        # res_layer['heatmap'] = one_hot
        res_layer['stage1_center'] = stage1_center

        res_layer["center"] = center_coords[:, :2] # no height info (x, y)
        res_layer["height"] = center_coords[:, [2]] # height info (z)
        res_layer["query_heatmap_score"] = one_hot

        # for k in res_layer.keys():
            # print(f'res_layer[{k}]', res_layer[k].shape)

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
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # center[:, 0, :] = center[:, 0, :] * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
        # center[:, 1, :] = center[:, 1, :] * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
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
            print(post_process_cfg)
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=ret_dict[k]['pred_scores'], box_preds=ret_dict[k]['pred_boxes'],
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=None
            )
            print('selected', selected.shape, ret_dict[k]['pred_boxes'].shape)
            ret_dict[k]['pred_boxes'] = ret_dict[k]['pred_boxes'][selected]
            ret_dict[k]['pred_scores'] = selected_scores
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'][selected]

            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1

        return ret_dict 
