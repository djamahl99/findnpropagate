import copy
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

from pcdet.utils.box_utils import boxes_to_corners_3d
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
from torchvision.ops import batched_nms
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from .transfusion_head_2D_proposals import PALETTE, draw_corners_on_image
from ..preprocessed_detector import PreprocessedDetector

from shapely.geometry import Polygon, Point

def draw_corners_on_be2v(corners, ax, color=(1, 1, 1), line_width=2, label='', max_num=500, tube_radius=None, linestyle='solid'):
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


    xt, yt = corners[:, 1].max(), corners[:, 0].max()

    if label != '':
        # ax.text(corners[6, 1] + 5, corners[6, 0] + 5, label, color=color)
        ax.text(xt + 5, yt + 5, label, color=(0, 0, 0))
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color, linestyle=linestyle)

        i, j = k + 4, (k + 1) % 4 + 4
        
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color, linestyle=linestyle)

        i, j = k, k + 4
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color, linestyle=linestyle)


    i, j = 0, 5
    ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color, linestyle=linestyle)

    i, j = 1, 4
    ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color, linestyle=linestyle)



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

# def cartesian_to_polar(xy):
#     x, y = xy

#     if x >= 0 and y >= 0:
#         # quandrant 1
#         return torch.atan(y / x)
#     elif x < 0 and y >= 0:
#         # quadrant 2
#         return torch.pi + torch.atan(y / x)
#     elif x < 0 and y < 0:
#         # quadrant 3
#         return torch.pi + torch.atan(y / x)
#     else:
#         return 2 * torch.pi + torch.atan(y / x)

def atan2_pos_angle(y, x):
    angles = torch.atan2(y, x)

    neg_angle_mask = (angles < 0)
    angles[neg_angle_mask] = torch.pi + torch.abs(-torch.pi - angles[neg_angle_mask])

    angles_over_pi = (angles > torch.pi)

    while angles_over_pi.sum() > 0:
        angles[angles_over_pi] -= torch.pi 

        angles_over_pi = (angles > torch.pi)

    return angles

def dist_point_to_line(p1, p2, p):
    frac = (p - p1) * (p2 - p1) / torch.norm(p2 - p1)

    return torch.norm((p - p1) - frac * (p2 - p1))

class TransFusionFrustumHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionFrustumHead, self).__init__()

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

        preds_path = self.model_cfg.PREDS_PATH
        camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.image_detector = PreprocessedDetector([preds_path + f"{cam_name}.json" for cam_name in camera_names])


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
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.x_size = x_size
        self.y_size = y_size

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

        pc_bev_pos = self.bev_pos.clone().reshape(-1, 2).cuda()
        pc_bev_pos[:, 0] = pc_bev_pos[:, 0] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        pc_bev_pos[:, 1] = pc_bev_pos[:, 1] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        # pc_bev_pos[:, 0] /= x_size
        # pc_bev_pos[:, 1] /= y_size
        # pc_bev_pos = pc_bev_pos * (self.point_cloud_max[:2] - self.point_cloud_min[:2]).unsqueeze(0) + self.point_cloud_min[:2]
        
        self.pc_bev_pos = pc_bev_pos
        self.bev_pos_long = (self.bev_pos.clone().reshape(-1, 2) - 0.5).long()

        print('bev pos range x ', pc_bev_pos[:, 0].min(), pc_bev_pos[:, 0].max())
        print('bev pos range y', pc_bev_pos[:, 1].min(), pc_bev_pos[:, 1].max())

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
        lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        frustum_min = torch.tensor(1.0, device='cuda')
        frustum_max = torch.tensor(100.0, device='cuda')

        
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        DEBUG = False

        if DEBUG:
            fig, ax = plt.subplots()
            fig.set_size_inches(5, 5)

            pts_bev = batch_dict['points'][..., 1:4].cpu()
            
            B, N = batch_dict['gt_boxes'].shape[0:2]

            gt_boxes = batch_dict['gt_boxes'].reshape(B*N, -1)
            gt_labels = gt_boxes[..., -1].long().cpu().numpy()
            gt_boxes = gt_boxes[gt_boxes[..., 0] == 0] # batch 0

            corners = boxes_to_corners_3d(gt_boxes)
            corners = corners.reshape(B*N, 8, 3)

        frustum_heatmap = torch.zeros((batch_size, self.num_classes, self.x_size, self.y_size), device='cuda')

        for b in range(batch_size):
            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            if cur_boxes.shape[0] >= self.num_proposals:
                print('there are more detected boxes than max proposals!', self.num_proposals)

            batch_box_idx = 0

            for c in range(6): # 6 cameras
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                # if batch_box_idx > self.num_proposals:
                    # break

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < 0.1:
                        continue

                    box = box.cuda()
                    # print('frustum_min', frustum_min.shape, box[0].shape)
                    xyzxyz = torch.cat([box[0][None], box[1][None], frustum_min[None], box[2][None], box[3][None], frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    bev_anchor = self.anchors[label, :2]
                    bev_min_dist = bev_anchor.min()
                    bev_max_dist = torch.norm(bev_anchor) / 2.0

                    frust_bev_box = torch.cat([frust_box[[2*i, 2*i+1], :2].mean(dim=0)[None, :] for i in range(4)], dim=0)

                    frust_center_close = frust_bev_box[:2, :].mean(dim=0)
                    frust_center_far = frust_bev_box[2:, :].mean(dim=0)

                    # frust_max_dist = frust_center_far.norm()
                    # frust_min_dist = frust_center_close.norm()
                    # frust_length = frust_max_dist - frust_min_dist

                    frust_left_close = frust_bev_box[0]
                    frust_right_close = frust_bev_box[1]

                    frust_left_vector = frust_bev_box[2]
                    frust_right_vector = frust_bev_box[3]
                    frust_center_vector = frust_center_far #- frust_center_close

                    frust_left_vector_norm = frust_left_vector / torch.norm(frust_left_vector, keepdim=True)
                    frust_right_vector_norm = frust_right_vector / torch.norm(frust_right_vector, keepdim=True)
                    frust_center_vector_norm = frust_center_vector / torch.norm(frust_center_vector, keepdim=True)

                    color=PALETTE[label % len(PALETTE)]

                    # ax.scatter([torch.cos(frust_left_angle).cpu() * 50], [torch.sin(frust_left_angle).cpu() * 50 ], marker='.', color=color)
                    # ax.scatter([torch.cos(frust_right_angle).cpu() * 50], [torch.sin(frust_right_angle).cpu() * 50 ], marker='o', color=color)
                    
                    bev_pos_norm = self.pc_bev_pos.clone().reshape(1, -1, 2)
                    bev_pos_norm = bev_pos_norm / torch.norm(bev_pos_norm, keepdim=True, dim=-1)

                    left_to_bev = bev_pos_norm.clone() - frust_left_vector_norm.reshape(1, 1, 2)
                    left_to_bev = left_to_bev / torch.norm(left_to_bev, keepdim=True, dim=-1)
                    right_to_bev = bev_pos_norm.clone() - frust_right_vector_norm.reshape(1, 1, 2)
                    right_to_bev = right_to_bev / torch.norm(right_to_bev, keepdim=True, dim=-1)
                    
                    # print('left to bev', left_to_bev.shape)
                    # print('right to bev', right_to_bev.shape)

                    left_to_bev = left_to_bev.reshape(-1, 2)
                    right_to_bev = right_to_bev.reshape(-1, 2)

                    cos_dir_lr = left_to_bev * right_to_bev
                    cos_dir_lr = cos_dir_lr.sum(dim=-1)
                    cos_dir_center = frust_center_vector_norm.reshape(-1, 1, 2) * bev_pos_norm
                    cos_dir_center = cos_dir_center.sum(dim=-1)
                    cos_dir_center = cos_dir_center.reshape(-1)

                    # print('cos dir', cos_dir_lr.min(), cos_dir_lr.max(), cos_dir_lr.shape)
                    # print('cos dir center', cos_dir_center.min(), cos_dir_center.max(), cos_dir_center.shape)

                    bev_in_range = torch.bitwise_and((cos_dir_lr < 0), (cos_dir_center > 0))


                    # if bev_valid_pts.shape[0] == 0:
                        # print('no valid pts?')
                    # else:
                        # print('VALID', frust_left_angle, frust_right_angle)

                    if DEBUG:
                        bev_valid_pts = self.pc_bev_pos[bev_in_range]
                        ax.scatter(bev_valid_pts[..., 0].cpu(), bev_valid_pts[..., 1].cpu(), color=color)
                        cls_name = self.all_class_names[label.item()]
                        draw_corners_on_image(frust_box.cpu(), ax, color=color, label=f'', linestyle='dotted')

                    for hm_coord in self.bev_pos_long[bev_in_range]:
                        y, x = hm_coord
                        frustum_heatmap[b, label.item(), x.item(), y.item()] = 1

                    batch_box_idx += 1

        if DEBUG:
            for b_idx in range(B*N):
                # print('corners sub', corners[b_idx])
                label = gt_labels[b_idx] - 1
                color = PALETTE[label % len(PALETTE)]
                cls_name = self.all_class_names[label.item()]


                chosen_corner = corners[b_idx, [0], :2]
                # find closest bev pos
                print('chosen corner', chosen_corner.shape, self.pc_bev_pos.shape)
                bev_idx = torch.cdist(chosen_corner, self.pc_bev_pos).argmin(dim=1)

                hm_coords = self.bev_pos_long[bev_idx]
                # print('hm coords', hm_coords)
                # heatmap2[label, hm_coords[:, 1], hm_coords[:, 0]] += 1

                draw_corners_on_image(corners[b_idx].cpu(), ax, color=(0, 0, 0), label=f'')

        if DEBUG:
            ax.invert_yaxis()
            plt.legend()
            plt.savefig(f'bev_frustums.png', bbox_inches='tight')
            plt.close()

        # frustum_heatmap = torch.softmax(frustum_heatmap, dim=1)

        inputs = batch_dict['spatial_features_2d']
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner

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
        res_layer['frustum_heatmap'] = frustum_heatmap

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

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # get pseudo label heatmap
        frustum_heatmap = pred_dicts['frustum_heatmap']

        # print('heatmap shape', heatmap.shape, 'frustum_heatmap', frustum_heatmap.shape)
        # heatmap = heatmap.reshape(-1, 1, heatmap.shape[-2], heatmap.shape[-1])
        # heatmap = heatmap.sum(dim=0, keepdim=True)
        # save_image(heatmap, 'true_heatmap.png', nrow=10, normalize=True)

        pred_hm = torch.softmax(pred_dicts["dense_heatmap"].detach().clone(), dim=1)
        save_image(pred_hm.reshape(-1, 1, heatmap.shape[-2], heatmap.shape[-1]), 'pred_hm.png', normalize=True, nrow=10)
        # exit()

        # set our pseudo labels as the heatmap for loss
        heatmap[:, 1:] = frustum_heatmap[:, 1:]
        # heatmap = 

        hm_view = heatmap.clone().reshape(-1, 1, heatmap.shape[-2], heatmap.shape[-1])
        save_image(hm_view, 'hm_view.png', nrow=10, normalize=True)

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        # ).mean()
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)

        one_hot_targets = torch.zeros(*list(labels.shape), self.num_classes+1, dtype=cls_score.dtype, device=labels.device)
        one_hot_targets.scatter_(-1, labels.unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., :-1]
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
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

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
