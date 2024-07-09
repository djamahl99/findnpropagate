import copy
import os
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from pcdet.models.backbones_image.clip_resnet import CLIPResNet
from pcdet.models.dense_heads.target_assigner.hungarian_assigner import HungarianAssigner3D
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils

from torchvision.utils import save_image
import torch.nn.functional as F

class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
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
    

class SeparateCLIPHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        text_metainfo_path = 'nuscenes_text.pkl'
        if os.path.exists(text_metainfo_path):
            text_metainfo = torch.load(text_metainfo_path)
            self.text_features = text_metainfo['text_features'].to('cuda')
            self.text_classes, self.text_dim = self.text_features.shape
            self.logit_scale = text_metainfo['logit_scale']

            print("Got stored text features", self.text_features.shape)
        else:
            raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
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
            # print("sep head task", cur_name)
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

            if cur_name == 'hm' and False:
                print("hm", ret_dict['hm'].shape)

                image_features = ret_dict['hm'] / torch.norm(ret_dict['hm'], dim=1, keepdim=True)
                text_features = self.text_features.to(image_features.dtype) #@ self.text_projection.to(x.dtype)

                print("image feat shape", image_features.shape)
                N, T, H, W = image_features.shape
                image_features = rearrange(image_features, 'N T H W -> (N H W) T')

                # cosine similarity as logits
                # logit_scale = self.clip.logit_scale.exp()
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                ret_dict['hm'] = rearrange(logits, '(N H W) C -> N C H W', N=N, H=H, W=W)
        
        return ret_dict

class CLIPSampleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.clip_head = CLIPResNet(dict(ATTNPOOLING=True, WEIGHTS='RN50'))

        # clip dim
        self.dim = self.clip_head.out_dim

        self.momentum = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=False)

        self.box_default_dim = nn.Parameter(torch.randn(1, self.dim), requires_grad=True)

        self.theta_rot = nn.Sequential(
            nn.Linear(9, 4),
            # nn.Linear(9, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(True),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(True),
            # nn.Linear(64, 4),
            nn.Tanh()
        )

        self.theta_rot[-2].weight.data.zero_()
        self.theta_rot[-2].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [512, 1408]

    def forward(self, batch_dict, preds_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        # x = batch_dict['image_fpn'] 
        # print("batch dict", batch_dict.keys())
        # print("preds_dict", [x.keys() for x in preds_dict])
        images = batch_dict['camera_imgs']

        batch_size = images.shape[0]

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # not in use
        points = batch_dict['points']

        # B x N x 1 x H x W
        # depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)

        num_boxes = sum(x['pred_boxes'].shape[0] for x in preds_dict)
        box_embeddings = self.box_default_dim.repeat(num_boxes, 1)
        box_i = 0

        theta_rot = self.theta_rot(torch.cat([x['pred_boxes'] for x in preds_dict]))

        for b in range(batch_size):
            print("boxes", preds_dict[b]['pred_boxes'].shape)
            cur_coords = preds_dict[b]['pred_boxes'][..., :3]
            print("number of boxes in this batch = ", cur_coords.shape[0])
            # print("box coords", cur_coords)
            # print("cur_coords", points[points[:,0] == b][:, 1:4])
            # exit()


            # batch_mask = points[:,0] == b
            # cur_coords = points[batch_mask][:, 1:4]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # do image aug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            # filter points outside of images
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            # print(f"on img [{b}]", on_img)
            # print(f"coords on img [{b}]", cur_coords[on_img].long())
            # print(f"coords on img [{b}]", cur_coords[on_img].shape)
            # print()
            # exit()
            for batched_box_i in range(preds_dict[b]['pred_boxes'].shape[0]):
                samples = []

                # for c in range(on_img.shape[0]):
                for c in self.image_order:
                    # masked_coords = cur_coords[c, on_img[c]] #.long()
                    if not on_img[c, batched_box_i]:
                        continue
                    
                    masked_coords = cur_coords[c, batched_box_i]

                    theta_y = masked_coords[..., [0]] / self.image_size[0]
                    theta_y = theta_y * 2 - 1
                    theta_x = masked_coords[..., [1]] / self.image_size[1]
                    theta_x = theta_x * 2 - 1

                    if theta_x.shape[0] == 0:
                        continue

                    theta_pos = torch.cat((theta_x, theta_y), dim=-1).reshape(-1, 2, 1)
                    # theta_eye = torch.eye(2).reshape(1, 2, 2).repeat(theta_pos.shape[0], 1, 1).to(theta_pos.device)
                    theta_eye = theta_rot[box_i].reshape(-1, 2, 2).repeat(theta_pos.shape[0], 1, 1)
                    # print('theta eye', theta_eye.shape, theta_rot.shape)
                    theta = torch.cat((theta_eye, theta_pos), dim=-1).to(images.device)
                    
                    for theta_i in range(theta.shape[0]):
                        # print(f"theta {theta_i}", theta[[theta_i]])
                        grid = F.affine_grid(theta[[theta_i]], [1, 3, 224, 224])
                        smp = F.grid_sample(images[b, [c]], grid)

                        samples.append(smp)

                        emb = self.clip_head(dict(camera_imgs=smp))['attn_features']

                        # EMA
                        box_embeddings[box_i] = box_embeddings[box_i].clone() * self.momentum + emb * (1 - self.momentum)
                        
                        
                box_i += 1
                
                if len(samples) > 0:
                    # print("num samples", len(samples))
                    samples = torch.cat(samples, dim=0)
                    save_image(samples, f'samples_proj_box_{box_i}.png')

        return box_embeddings
            #     masked_dist = dist[c, on_img[c]]
            #     depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

class CenterHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        text_metainfo_path = 'nuscenes_text.pkl'
        if os.path.exists(text_metainfo_path):
            text_metainfo = torch.load(text_metainfo_path)
            self.text_features = text_metainfo['text_features'].to('cuda')
            self.text_classes, self.text_dim = self.text_features.shape
            self.logit_scale = text_metainfo['logit_scale']

            print("Got stored text features", self.text_features.shape)
        else:
            raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")
        
        head_module = SeparateCLIPHead if self.model_cfg.get('SEPARATECLIPHEAD', False) else SeparateHead

        self.clip_sample_head = CLIPSampleHead()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            # cur_head_dict['hm'] = dict(out_channels=self.text_dim, num_conv=self.model_cfg.NUM_HM_CONV)
            # cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            cur_head_dict['hm'] = dict(out_channels=1, num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                head_module(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )

        predict_boxes_when_training = True
        assert predict_boxes_when_training, "no pred while train :'("

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())
        self.add_module('box_ce_loss_func', nn.CrossEntropyLoss())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        pred_dicts_boxes = self.forward_ret_dict['pred_dicts_boxes']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            with torch.no_grad():
                # true as one class
                target_dicts['heatmaps'][idx] = torch.clamp(target_dicts['heatmaps'][idx].clone().sum(dim=1, keepdim=True), 0.0, 1.0)

                # pred_hm = pred_dict['hm'].clone().sum(dim=1, keepdim=True)
                N, C, H, W = pred_dict['hm'].shape
                pred_hm = pred_dict['hm'].clone().reshape(N*C, 1, H, W)
                # true_hm = target_dicts['heatmaps'][idx].clone().sum(dim=1, keepdim=True)
                true_hm = target_dicts['heatmaps'][idx].clone().reshape(N*C, 1, H, W)



                save_image(pred_hm, 'pred_hm.png', nrow=C, scale_each=True)
                save_image(true_hm, 'true_hm.png', nrow=C, scale_each=True)

            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']



            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            target_classes = target_boxes[..., -1].long()
            print("target_classes", target_classes)
            print("target_classes.shape", target_classes.shape)
            print('target_boxes', target_boxes.shape)
            # print('pred_boxes', pred_boxes.shape)
            # exit()

            # bboxes = torch.cat([x['pred_boxes'][None] for x in pred_dicts]).reshape(-1, 9)
            print("pred dicts keys", pred_dicts[idx].keys())
            bboxes = pred_dicts_boxes[idx]['pred_boxes']
            logits = pred_dicts_boxes[idx]['logits']
            gt_bboxes = target_boxes[..., :9] #.reshape(-1, 9)
            print("bboxes,gt_bboxes", bboxes.shape, gt_bboxes.shape)
            gt_labels = gt_bboxes[..., -1].long() #.reshape(-1).long()
            # gt_labels = data_dict['gt_boxes'][..., -1].reshape(-1).long()
            print('gt labels', gt_labels.shape, gt_labels)
            target_inds, _ = self.bbox_assigner.assign(bboxes, gt_bboxes, gt_labels, logits, self.point_cloud_range)

            print("target indx", target_inds)
            print("target indx shape", target_inds.shape)

            print("logits / gtlabels", logits.shape, gt_labels.shape)
            print("logits / gtlabels", logits.shape, gt_labels.shape)
            cls_loss = self.box_ce_loss_func(logits, gt_labels[target_inds])

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            self.forward_ret_dict['target_dicts'] = target_dict

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:

            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

            self.forward_ret_dict['pred_dicts_boxes'] = pred_dicts
            box_embs = self.clip_sample_head(data_dict, pred_dicts)

            # CLIP 
            box_features = box_embs / torch.norm(box_embs, dim=1, keepdim=True)
            text_features = self.text_features.to(box_features.dtype) #@ self.text_projection.to(x.dtype)

            # print("box_features shape", box_features.shape)

            # cosine similarity as logits
            # logit_scale = self.clip.logit_scale.exp()
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * box_features @ text_features.t()

            logits = logits.unsqueeze(0).permute(0, 2, 1)
            # print("logits", logits.shape)

            # print("pred dicts", self.forward_ret_dict['pred_dicts'])
            print("pred dicts 0", self.forward_ret_dict['pred_dicts'][0].keys())
            # print("pred dicts 0", self.forward_ret_dict['pred_dicts'][0]['center'])
            self.forward_ret_dict['pred_dicts_boxes'][0]['logits'] = logits

        return data_dict
