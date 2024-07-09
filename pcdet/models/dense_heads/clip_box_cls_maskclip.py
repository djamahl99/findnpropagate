import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
from torchvision import transforms as T

from pcdet.utils import common_utils
from typing import Dict, List, Tuple, Union
from shapely.geometry import MultiPoint, box, LineString, Polygon

from PIL import Image
import clip
from clip.model import CLIP
from clip import available_models, tokenize
from tqdm import tqdm
from pcdet.utils.box_utils import boxes_to_corners_3d

from prompts import imagenet_templates

# https://github.com/open-mmlab/mmdetection3d/blob/0f9dfa97a35ef87e16b700742d3c358d0ad15452/tools/dataset_converters/nuscenes_converter.py#L541
def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)

        if isinstance(img_intersection, LineString):
            intersection_coords = np.array(
                [coord for coord in img_intersection.coords])
        elif isinstance(img_intersection, Polygon):
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords])
        else:
            print(f'img_intersection:{img_intersection} should be Polygon or LineString!')
            return None

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

class DenseAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, text_features=None, logit_scale=None, refine=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        # # text embeddings
        # text_metainfo_path = '/home/uqdetche/OpenPCDet/tools/nuscenes_text.pkl'
        # if os.path.exists(text_metainfo_path):
        #     text_metainfo = torch.load(text_metainfo_path)
        #     self.text_features = text_metainfo['text_features'].to('cuda', dtype=torch.float32)
        #     self.text_classes, self.text_dim = self.text_features.shape
        #     self.logit_scale = torch.tensor(text_metainfo['logit_scale'], device='cuda')

        #     print("Got stored text features", self.text_features.shape)
        # else:
        #     raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")

        self.text_features = text_features
        self.logit_scale = logit_scale
        self.refine = refine
        self.pd_thresh = 0

    def forward(self, x, dense=True):
        if not dense:
            x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            
            x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

            x, _ = F.multi_head_attention_forward(
                query=x[:1], key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
            )
            x = x.squeeze(0)
            x = x / x.norm(dim=1, keepdim=True)
            x = x[..., None, None]
            output = F.conv2d(x, self.text_features[:, :, None, None])

            output = F.softmax(output*self.logit_scale, dim=1)

            return output

        else:
            # dense mode -> use conv
            q = F.conv2d(input=x, weight=self.q_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.q_proj.bias)
            k = F.conv2d(input=x, weight=self.k_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.k_proj.bias)
            q = torch.flatten(q, start_dim=2).transpose(-2, -1)
            k = torch.flatten(k, start_dim=2).transpose(-2, -1)
            v = F.conv2d(input=x, weight=self.v_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.v_proj.bias)
            feat = F.conv2d(input=v, weight=self.c_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.c_proj.bias)

            feat = feat / feat.norm(dim=1, keepdim=True)
            # return feat
            self.text_features = self.text_features.to(dtype=feat.dtype)
            output = F.conv2d(feat, self.text_features[:, :, None, None])

            if self.refine:
                output = self.refine_output(output, k)
            else:
                output = F.softmax(output*self.logit_scale, dim=1)

            return output
        
    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output
    
class CLIPTextEnsembling(nn.Module):
    def __init__(self, clip: CLIP) -> None:
        super().__init__()

        self.clip = clip

        # from openscene paper
        self.openscene_nusc_labels = {
            'barrier': ['barrier', 'barricade'],
            'bicycle': ['bicycle'],
            'bus': ['bus'],
            'car': ['car', 'van'], # added van
            'construction_vehicle': ['bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck'],
            'motorcycle': ['motorcycle'],
            'pedestrian': ['pedestrian', 'person'],
            'traffic_cone': ['traffic cone'],
            'trailer': ['trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container'],
            'truck': ['truck']
        }

        embs1 = self.imagenet_template_ensembling(self.openscene_nusc_labels.keys())

        # embs = self.threeways_variant_ensembling(self.openscene_nusc_labels.keys())
        # print('embs dropout', embs.shape)

        embs2 = self.openscene_category_ensembling(self.openscene_nusc_labels.keys())

        sims = torch.cosine_similarity(embs1, embs2)

        print('Text similarity between imagenet templates + openscene category ensembling', sims)

    # CLIP paper
    @torch.no_grad()
    def imagenet_template_ensembling(self, category_list):
        templates = imagenet_templates
        texts = [template.format(cetegory) for cetegory in category_list for template in templates] #format with class
        # exit(d)
        texts = tokenize(texts, context_length=77, truncate=True).to(self.clip.positional_embedding.device)
        class_embeddings = []
        cursor = 0
        step = 1 
        while cursor <= len(texts):
            class_embeddings.append(self.clip.encode_text(texts[cursor:cursor + step]))
            cursor += step
        class_embeddings = torch.cat(class_embeddings)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unflatten(0, (len(category_list), len(templates)))
        class_embedding = class_embeddings.mean(dim=1)
        class_embedding = class_embedding / class_embedding.norm(dim=-1, keepdim=True)
        return class_embedding
    
    # threeways paper
    # @torch.no_grad()
    # def threeways_variant_ensembling(self, category_list, variants=64):
    #     # set dropout to train

    #     modules_list = []
    #     modules_list.extend(list(self.clip.transformer.modules()))

    #     found_dropout = False

    #     i = 0
    #     while i < len(modules_list):
    #         m = modules_list[i]

    #         modules_list.extend(list(m.children()))

    #         if 'Dropout' in m._get_name():
    #             print('dropout layer=>', m)
    #             found_dropout = True
    #             m.train()
    #         else:
    #             m.eval()

    #         i += 1

    #     assert found_dropout, 'could not find dropout layer!'

    #     class_embeddings = []

    #     for cls_name in category_list:
    #         variant_embeddings = []
    #         texts = tokenize([f'a photo of a {cls_name}'], context_length=77, truncate=True).to(self.clip.positional_embedding.device)

    #         for i in range(variants):
    #             variant_embeddings.append(self.clip.encode_text(texts))

    #         variant_embeddings = torch.cat(variant_embeddings)
    #         variant_embeddings = variant_embeddings / variant_embeddings.norm(dim=-1, keepdim=True)                
    #         variant_embeddings = variant_embeddings.mean(dim=0)
    #         variant_embeddings = variant_embeddings / variant_embeddings.norm(dim=-1, keepdim=True)                

    #         class_embeddings.append(variant_embeddings.unsqueeze(0))

    #     class_embeddings = torch.cat(class_embeddings)

    #     for m in self.clip.transformer.modules():
    #         m.eval()

    #     return class_embeddings

    @torch.no_grad()
    def openscene_category_ensembling(self, category_list):
        class_embeddings = []
        # for (cat, predefined_labels) in self.openscene_nusc_labels.items():
        for cat in category_list:
            assert cat in self.openscene_nusc_labels.keys(), f'{cat} not in OpenScene predefined label dict!'
                
            predefined_labels = self.openscene_nusc_labels[cat]

            texts = [f'a photo of a {label}' for label in predefined_labels]
            texts = tokenize(texts, context_length=77, truncate=True).to(self.clip.positional_embedding.device)

            curr_embeddings = self.clip.encode_text(texts)

            curr_embeddings = curr_embeddings / curr_embeddings.norm(dim=-1, keepdim=True)
            curr_embeddings = curr_embeddings.mean(dim=0)
            curr_embeddings = curr_embeddings / curr_embeddings.norm(dim=-1, keepdim=True)

            class_embeddings.append(curr_embeddings.unsqueeze(0))
            
        class_embeddings = torch.cat(class_embeddings)
        print('class_embeddings', class_embeddings.shape)
        return class_embeddings
    
    @torch.no_grad()
    def no_ensembling(self, category_list):
        text = clip.tokenize([f'a photo of a {x}' for x in category_list]).to('cuda')
        text_features = self.clip.encode_text(text)

        return text_features

class MaskCLIP(nn.Module):
    """ CLIP ResNet visual model.

    Args:
        WEIGHTS (str): which pretrained CLIP backbone to load
    """
    
    def __init__(self, clip: CLIP, text_features: Tensor) -> None:
        super(MaskCLIP, self).__init__()

        self.out_dim = clip.visual.output_dim

        clip = clip.to(dtype=torch.float32)

        self.stem = nn.Sequential(
            clip.visual.conv1,
            clip.visual.bn1,
            clip.visual.relu1,

            clip.visual.conv2,
            clip.visual.bn2,
            clip.visual.relu2,
            
            clip.visual.conv3,
            clip.visual.bn3,
            clip.visual.relu3,
            
            clip.visual.avgpool
        )

        self.layers = nn.ModuleList([
            clip.visual.layer1,
            clip.visual.layer2,
            clip.visual.layer3,
            clip.visual.layer4,
        ])

        attnpool = clip.visual.attnpool
        spacial_dim = np.sqrt(attnpool.positional_embedding.shape[0] - 1)
        spacial_dim = int(spacial_dim)
        embed_dim = attnpool.k_proj.in_features
        output_dim = attnpool.c_proj.out_features

        # dense atten pooling (to get segmentation map)
        self.attnpool = DenseAttentionPool2d(spacial_dim, embed_dim, attnpool.num_heads, output_dim, text_features=text_features, logit_scale=clip.logit_scale.data)
        self.attnpool.load_state_dict(clip.visual.attnpool.state_dict())

    def forward(self, x, dense=True):
        x = x.type(self.stem[0].weight.dtype)
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
        
        out = self.attnpool(x, dense=dense)

        return out

class CLIPBoxClassificationMaskCLIP(nn.Module):
    def __init__(self, image_size=[900, 1600], clip_model="RN50x4", ensembling=None, one_hot=False) -> None:
        super().__init__()

        self.one_hot = one_hot

        self.image_order = [2, 0, 1, 5, 3, 4] # for plotting
        self.image_size = image_size

        self.all_class_names = ['car','truck', 'construction vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic cone']

        model, preprocess = clip.load(clip_model, device='cuda')
        self.clip = model

        # ensembling text prompts
        # self.text_features = self.forward_feature(self.all_class_names)
        
        self.texts_ensembler = CLIPTextEnsembling(model)
        self.text_features = None

        if ensembling == 'template':
            self.text_features = self.texts_ensembler.imagenet_template_ensembling(self.all_class_names)
        elif ensembling == 'openscene':
            self.text_features = self.texts_ensembler.openscene_category_ensembling(self.all_class_names)
        elif ensembling is None:
            self.text_features = self.texts_ensembler.no_ensembling(self.all_class_names)
        else:
            print('invalid ensembling')
        
        self.maskclip = MaskCLIP(model, self.text_features)

        self.vis_images = None

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        cur_coords = points.clone()

        # camera_intrinsics = batch_dict['camera_intrinsics']
        # camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        cur_img_aug_matrix = img_aug_matrix[batch_idx, [cam_idx]]
        cur_lidar_aug_matrix = lidar_aug_matrix[batch_idx]
        cur_lidar2image = lidar2image[batch_idx, [cam_idx]]

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
        cur_coords = cur_coords[..., [1, 0]]

        # filter points outside of images
        # on_img = (
        #     (cur_coords[..., 1] < self.image_size[0])
        #     & (cur_coords[..., 1] >= 0)
        #     & (cur_coords[..., 0] < self.image_size[1])
        #     & (cur_coords[..., 0] >= 0)
        # )

        return cur_coords
    
    def get_seg_mask_coords(self, cur_images: Tensor, seg_masks: Tensor):
        h, w = cur_images.shape[-2:]
        sh, sw = seg_masks.shape[-2:]

        xs = torch.linspace(0, h-1, steps=sh, dtype=torch.long, device=seg_masks.device)
        ys = torch.linspace(0, w-1, steps=sw, dtype=torch.long, device=seg_masks.device)
    
        xs, ys = torch.meshgrid((xs, ys))

        xys = torch.cat([xs.unsqueeze(-1), ys.unsqueeze(-1)], dim=-1)

        return xys

        seg_xidx = torch.linspace(0, sh-1, steps=sh, dtype=torch.long, device=seg_masks.device)
        seg_yidx = torch.linspace(0, sw-1, steps=sw, dtype=torch.long, device=seg_masks.device)

        idx_xs, idx_ys = torch.meshgrid((seg_xidx, seg_yidx))
        xys_idx = torch.cat([idx_xs.unsqueeze(-1), idx_ys.unsqueeze(-1)], dim=-1)


        return xys, xys_idx

    def vis_seg_masks(self, images, seg_masks, filename='clip_vis_mask.png', class_colors=None, save=True):
        if class_colors is None:
            class_colors = [(220, 20, 60), (119, 11, 32), (0, 0, 0), (0, 0, 0),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 0),
                (0, 0, 0), (250, 170, 30)]
            class_colors = [[y/255 for y in x] for x in class_colors]


        # class_colors = [[(255, 0, 0)]  , (0,)]
# ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            #   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        class_colors = torch.tensor(class_colors, dtype=torch.float32, device=images.device)
        

        vh, vw = seg_masks.shape[-2:]
        ih, iw = images.shape[-2:]

        seg_masks = seg_masks.reshape(-1, seg_masks.shape[1], vh*vw)
        out_max = torch.max(seg_masks, dim=1)
        out_argmax = out_max.indices

        # colorize
        out_rgb = class_colors[out_argmax]
        out_rgb[out_max.values < 1/len(class_colors), :] = 0.0 # background
        # print('out_max.values', out_max.values.max())
        # out_rgb[out_max.values < 0.5, :] = 0.0 # background
        
        # brightness by confidence?
        # out_rgb = out_rgb * out_max.values.unsqueeze(-1)

        out_rgb = out_rgb.reshape(-1, vh, vw, 3).permute(0, 3, 1, 2)

        resize_seg = T.Resize((ih, iw), interpolation=T.InterpolationMode.NEAREST)        
        out_rgb = resize_seg(out_rgb)

        vis_image = out_rgb * 0.5 + images * 0.5
        vis_image = vis_image[[2, 0, 1, 5, 3, 4]]
        if save:
            save_image(vis_image, filename)
        
        return vis_image

    def forward(self, batch_dict, pred_dicts, keep_images=False, class_colors=None):
        """
        Args:
            batch_dict:
                batch information containing, e.g. images, camera_intrinsics etc keyss

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """

        batch_size = batch_dict['batch_size']
        images = batch_dict['camera_imgs']

        theta = torch.eye(2, 3).unsqueeze(0)
        self.unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]).cuda() # -1, 1 grid

        seg_image_coords = None

        for b in range(batch_size):
            cur_pred_dict = pred_dicts[b]
            cur_images = images[b]

            # run maskclip on the camera images
            seg_masks = self.maskclip(cur_images)
            # print('seg_masks', seg_masks.min(), seg_masks.max())

            # debugging
            # self.vis_seg_masks(cur_images, seg_masks, f'clip_vis_mask_{b}.png')


            # coords of each seg superpixel
            if seg_image_coords is None:
                seg_image_coords = self.get_seg_mask_coords(cur_images, seg_masks)


            cur_boxes = cur_pred_dict['pred_boxes']
            N = cur_boxes.shape[0]

            if N == 0:
                print('no boxes for clip!')
                # print(cur_pred_dict)
                continue

            box_probs = torch.zeros((N, 10), device='cuda', dtype=torch.half) # 6 cameras, 10 classes
            box_pixel_sum = torch.zeros((N,), device='cuda', dtype=torch.long)
            box_cam_mask = torch.zeros((N, 6), device='cuda', dtype=torch.bool)

            # box number
            cur_idx = torch.arange(0, N).reshape(N, 1).repeat(1, 8).reshape(N*8)
            corners = boxes_to_corners_3d(cur_boxes)
            corners = corners.reshape(-1, 3)

            for c in self.image_order:
                all_coords = self.project_to_camera(batch_dict, corners, b, c).reshape(-1, 2).long()

                for idx in range(N):
                    coord_mask = (cur_idx == idx)
                    box_cam_coords = all_coords[coord_mask]

                    x1, x2 = box_cam_coords[..., 1].min(), box_cam_coords[..., 1].max()
                    y1, y2 = box_cam_coords[..., 0].min(), box_cam_coords[..., 0].max()

                    x1 = torch.clamp(x1, 0, self.image_size[1])
                    x2 = torch.clamp(x2, 0, self.image_size[1])
                    y1 = torch.clamp(y1, 0, self.image_size[0])
                    y2 = torch.clamp(y2, 0, self.image_size[0])

                    # this box occurs on this camera
                    box_cam_mask[idx, c] = 1

                    # find coords that fit
                    # seg_coords_mask = (seg_image_coords[..., 0] <= x2) & (seg_image_coords[..., 0] >= x1) & (seg_image_coords[..., 1] <= y2) & (seg_image_coords[..., 1] >= y1)
                    seg_coords_mask = (seg_image_coords[..., 1] <= x2) & (seg_image_coords[..., 1] >= x1) & (seg_image_coords[..., 0] <= y2) & (seg_image_coords[..., 0] >= y1)

                    if seg_coords_mask.sum() == 0: 
                        continue

                    segs = seg_masks[c, :, seg_coords_mask]


                    # debugging
                    # self.vis_seg_masks(cur_images, seg_masks, f'clip_vis_mask_{idx}_{c}.png')

                    seg_sum = segs.sum(dim=-1)
                    # print('seg_sum', seg_sum)
                    # # box_probs[idx] += segs.sum(dim=-1)
                    if self.one_hot:
                        seg_max = torch.max(segs, dim=0)
                        ohs = F.one_hot(seg_max.indices, num_classes=len(self.all_class_names)).cuda().float()
                        ohs *= seg_max.values.unsqueeze(-1)
                        # print('seg_mean', seg_mean)
                        # print('ohs', ohs)
                        box_probs[idx] += ohs.sum(dim=0)
                    else:
                        box_probs[idx] += seg_sum
                    box_pixel_sum[idx] += segs.shape[-1]

                    # seg_thresh = (segs > 0.5).sum(dim=-1)
                    # print('seg_thresh', seg_thresh)
                    # # exit()
                    # box_probs[idx] += (segs > 0.5).sum(dim=-1)
                    # box_pixel_sum[idx] += (segs > 0.5).sum(dim=-1).sum()

            box_sums = box_cam_mask.sum(dim=1)
            
            # mean over the camera images (that this box actually showed in)
            # box_probs_mean = box_probs.sum(dim=1) / box_cam_mask.sum(dim=-1).unsqueeze(1)
            box_probs_mean = box_probs / box_pixel_sum.unsqueeze(1)
            # print('box_probs means', box_probs_mean)
            # print('box probs', box_probs)

            # print('original pred_labels', pred_dicts[b]['pred_labels'])

            pred_max = torch.max(box_probs_mean, dim=-1, keepdim=True)#.cpu()
            pred_labels = pred_max.indices.cpu()
            pred_scores = pred_max.values.cpu()
            # pred_labels = torch.argmax(box_probs_mean, dim=-1, keepdim=True).cpu()
            
            # print('pred_labels', pred_labels + 1)
            # pred_labels = pred_labels.flatten().cpu()

            pred_dicts[b]['pred_labels'] = pred_labels + 1 # 0 -> bg, 1 -> car, ...
            pred_dicts[b]['pred_scores'] = pred_scores # 0 -> bg, 1 -> car, ...

            # print('boxes not on any camera: ', (box_sums == 0).sum())


            if keep_images:
                self.vis_images = self.vis_seg_masks(cur_images, seg_masks, save=True)
            
        return pred_dicts
    
if __name__ == '__main__':
    clip_cls_maskclip = CLIPBoxClassificationMaskCLIP()

    texts_ensembler = CLIPTextEnsembling(clip_cls_maskclip.clip)

    