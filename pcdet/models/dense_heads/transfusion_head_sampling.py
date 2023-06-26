import copy
import os
from typing import List
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from ..backbones_image.clip_resnet import CLIPResNet

from torchvision.utils import save_image
from torchvision import transforms as IT
from PIL import Image
from einops import rearrange
from ...utils import box_utils
torch.autograd.set_detect_anomaly(True)
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


class ImageSampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.transform = IT.Compose([
            IT.Resize((448)),
            IT.ToTensor(),
            IT.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.output_size = [3, 224, 224]

        # self.image_order = [4, 2, 0, 1, 5, 3]
        self.image_order = [2, 0, 1, 5, 3, 4]

    def load_images(self, img_paths):
        # load and concatenate along width
        images = []
        for batch in img_paths:
            # print("paths", batch)
            # print("paths ordered", batch[self.image_order])
            # for path in batch:
            for i in self.image_order:
                path = batch[i]
                path = f"/home/uqdetche/mmdetection3d/data/nuscenes/{path}"
                img = Image.open(path)
                img = self.transform(img)
                images.append(img.unsqueeze(0))

        samples = torch.cat(images, dim=-1)
        return samples

    def forward(self, batch_dict, theta: Tensor):
        """

        Args:
            x (_type_): _description_
            img_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        # print(batch_dict.keys())
        images = self.load_images(batch_dict['image_paths'])
        images = images.to(theta.device)
        # save_image(images, 'loaded_images.png')

        # x = batch_dict['camera_imgs']
        # print("images shape", x.shape)
        # if len(x.shape) == 5:
        #     B, N, C, H, W = x.size()
        #     # x = x.view(B * N, C, H, W)
        #     # x = x.view(B * N, C, H, W)
        #     x = x[:, self.image_order].permute(1, 0, 2, 3, 4)
        #     # x = x.reshape(B, C, H, W*N)
        #     # concatenate for width
        #     x = torch.cat([i for i in x], dim=-1)
        #     save_image(x, 'images.png')
        # elif len(x.shape) == 4:
        #     BN, C, H, W = x.size()
        #     B = BN // 6
        #     N = 6


        # images = x
        # theta = theta.view(-1, 2, 3)
        N = theta.shape[0]
        # images = images.repeat((N, 1, 1, 1))


        samples = []
        for i in range(N):
            grid = F.affine_grid(theta[[i]], [1, 3, 224, 224])
            smp = F.grid_sample(images, grid)
            samples.append(smp)

        samples = torch.cat(samples, dim=0)

        # save_image(samples, 'samples.png')
        save_image(samples[[0]], 'sample0.png')

        return samples
    

class CLIPSampleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.clip_head = CLIPResNet(dict(ATTNPOOLING=True, WEIGHTS='RN50'))

        # clip dim
        self.dim = self.clip_head.out_dim

        self.momentum = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=False)

        self.box_default_dim = nn.Parameter(torch.randn(1, self.dim), requires_grad=True)

        # self.theta_rot = nn.Sequential(
        #     nn.Linear(9, 4),
        #     # nn.Linear(9, 64),
        #     # nn.LayerNorm(64),
        #     # nn.ReLU(True),
        #     # nn.Linear(64, 64),
        #     # nn.LayerNorm(64),
        #     # nn.ReLU(True),
        #     # nn.Linear(64, 4),
        #     nn.Tanh()
        # )

        # self.theta_rot[-2].weight.data.zero_()
        # self.theta_rot[-2].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [512, 1408]

    def forward(self, batch_dict, thetas, query_pos):
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
        # points = batch_dict['points']

        # B x N x 1 x H x W
        # depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)

        print("query pos shape", query_pos.shape, thetas.shape)
        num_boxes = query_pos.shape[0] * query_pos.shape[1]
        box_embeddings = self.box_default_dim.repeat(num_boxes, 1)
        box_i = 0

        for b in range(batch_size):
            # print("queries", query_pos[b])
            # theta_rot = thetas[b]

            # print("thetas", theta_rot)
            cur_coords = query_pos[b, :, :3]
            # print("number of boxes in this batch = ", cur_coords.shape[0])
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
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :].clone(), 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :].clone()

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
            for batched_box_i in range(query_pos.shape[1]):
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
                    theta_eye = thetas[b, batched_box_i].reshape(-1, 2, 2).repeat(theta_pos.shape[0], 1, 1)
                    print('theta eye', theta_eye, thetas[b, batched_box_i])
                    # print('theta rot', theta_rot.shape)
                    theta = torch.cat((theta_eye, theta_pos), dim=-1).to(images.device)
                    
                    for theta_i in range(theta.shape[0]):
                        # print(f"theta {theta_i}", theta[[theta_i]])
                        grid = F.affine_grid(theta[[theta_i]], [1, 3, 224, 224])
                        smp = F.grid_sample(images[b, [c]], grid)

                        # samples.append(smp)

                        emb = self.clip_head(dict(camera_imgs=smp))['attn_features']

                        # EMA
                        box_embeddings[box_i] = box_embeddings[box_i].clone() * self.momentum + emb * (1 - self.momentum)
                        
                        
                box_i += 1
                
                # if len(samples) > 0:
                    # print("num samples", len(samples))
                    # samples = torch.cat(samples, dim=0)
                    # save_image(samples, f'samples_proj_box_{box_i}.png')

        return box_embeddings

class TransFusionHeadSampling(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHeadSampling, self).__init__()

        # override
        print("class names", class_names)
        # num_class = 1
        self.class_names = class_names

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


        text_metainfo_path = 'nuscenes_text.pkl'
        if os.path.exists(text_metainfo_path):
            text_metainfo = torch.load(text_metainfo_path)
            self.text_features = text_metainfo['text_features'].to('cuda')
            self.text_classes, self.text_dim = self.text_features.shape
            self.logit_scale = text_metainfo['logit_scale']

            print("Got stored text features", self.text_features.shape)
        else:
            raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")
        
        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)
        layers = []
        layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=self.text_dim,kernel_size=3,padding=1))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        # heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        # heads.pop('heatmap')
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        # sampling grid preds
        # self.image_sampler = ImageSampler()
        # self.clip_head = CLIPResNet(dict(ATTNPOOLING=True, WEIGHTS='RN50'))
        self.clip_sample_head = CLIPSampleHead()
        self.sampling_ffn = nn.Sequential(
            nn.Linear(hidden_channel, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 2)
        )
        
        # guessed :)
        self.register_buffer('image_face_angles', torch.tensor([3*np.pi/4, np.pi/2, np.pi/4, 7*np.pi/4, 3*np.pi/2, 5*np.pi/4], dtype=torch.float32).unsqueeze(0))

        self.bev_to_3d = nn.Linear(2, 3)
        self.register_buffer('image_offsets', torch.linspace(-1, 1, 6))

        self.init_weights()

        self.sampling_ffn[2].weight.data.zero_()
        # self.sampling_ffn[2].bias.data.copy_(torch.tensor([448/4776, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.sampling_ffn[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # self.sampling_ffn[2].bias.data.copy_(torch.tensor([448/4776, 0, 0, 1], dtype=torch.float))
        self.sampling_ffn[2].bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))

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

    def predict(self, batch_dict):
        inputs = batch_dict['spatial_features_2d']

        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)
        N, T, H, W = dense_heatmap.shape
        dense_heatmap = rearrange(dense_heatmap, 'N T H W -> (N H W) T')

        dense_heatmap = dense_heatmap / torch.norm(dense_heatmap, dim=1, keepdim=True)
        # dense_heatmap_embed = dense_heatmap.clone()
        text_features = self.text_features.to(dense_heatmap.dtype)
        dense_heatmap = dense_heatmap @ text_features.t()
        dense_heatmap = rearrange(dense_heatmap, '(N H W) C -> N C H W', N=N, H=H, W=W).contiguous()

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

        thetas = self.sampling_ffn(query_feat.permute(0, 2, 1)) #.reshape(-1, 2, 2)
        theta_pos = self.bev_to_3d(query_pos) #.reshape(-1, 2, 1)

        # sampled_images = self.image_sampler(batch_dict, thetas)
        # image_features = self.clip_head(dict(camera_imgs=sampled_images))['attn_features']
        # print("query_pos", query_pos)
        image_features = self.clip_sample_head(batch_dict, thetas, theta_pos)


        # CLIP
        image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
        text_features = self.text_features.to(image_features.dtype) #@ self.text_projection.to(x.dtype)

        # cosine similarity as logits
        # logit_scale = self.clip.logit_scale.exp()
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        num_classes = logits.shape[-1]

        logits = logits.reshape(batch_size, self.num_proposals, num_classes).permute(0, 2, 1)


        # print("logits softmax", torch.softmax(logits, dim=-1))
        # print("logits argmax", torch.argmax(logits, dim=1))
        
        res_layer = self.prediction_head(query_feat)
        assert 'heatmap' not in res_layer

        res_layer['heatmap'] = logits

        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        # print("center ", res_layer["center"].shape)
        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, 1, -1),
            dim=-1,
        )

        # cl

        res_layer["dense_heatmap"] = dense_heatmap

        # print("query_heatmap_score", res_layer['query_heatmap_score'].shape)
        # print("dense_heatmap", res_layer['dense_heatmap'].shape)

        return res_layer

    def forward(self, batch_dict):
        # print("batch_dict", batch_dict.keys())
        # print("camera images", batch_dict['camera_imgs'].shape)
        # print("lidar2image", batch_dict['lidar2image'].shape)

        # print('paths', batch_dict['image_paths'])

        # B, N, C, H, W = batch_dict['camera_imgs'].shape
        
        # feats = batch_dict['spatial_features_2d']
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

        # class agnostic
        # heatmap = heatmap.sum(dim=1, keepdim=True)
        # heatmap = torch.clamp(heatmap, 0.0, 1.0)

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        # save_image(clip_sigmoid(pred_dicts["dense_heatmap"].detach().clone().sum(dim=1, keepdim=True)), 'dense_heatmap.png')

        with torch.no_grad():
            N, C, H, W = heatmap.shape
            print("pred hm", pred_dicts["dense_heatmap"].shape, 'true', heatmap.shape)
            save_image(clip_sigmoid(pred_dicts["dense_heatmap"].detach().clone().reshape(N*C, 1, H, W)), 'pred_hm.png', nrow=C, scale_each=True)
            save_image(heatmap.detach().clone().reshape(N*C, 1, H, W), 'true_hm.png', nrow=C, scale_each=True)
            # save_image(heatmap.detach().clone().sum(dim=1, keepdim=True), 'heatmap.png')

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
