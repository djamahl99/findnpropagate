import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage

from pcdet.utils import common_utils
from typing import Dict, List, Tuple, Union
from shapely.geometry import MultiPoint, box, LineString, Polygon

from PIL import Image
import clip
from clip import available_models, tokenize
from tqdm import tqdm
from pcdet.utils.box_utils import boxes_to_corners_3d
from pcdet.models.dense_heads.clip_box_cls_maskclip import CLIPTextEnsembling


# from prompts import imagenet_templates

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
    print('FAIL')
    exit()
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

class CLIPBoxClassification(nn.Module):
    def __init__(self, image_size=[900, 1600], clip_model="ViT-L/14", ensembling=None) -> None:
        super().__init__()

        self.image_order = [2, 0, 1, 5, 3, 4] # for plotting
        self.image_size = image_size

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

        print('clip_model', clip_model)
        print(clip.available_models())
        model, preprocess = clip.load(clip_model, device='cuda')
        self.clip = model

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

        self.min_crop_size = 64

        theta = torch.eye(2, 3).unsqueeze(0)
        self.unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]).cuda() # -1, 1 grid

    def clip_coords(self, masked_coords):
        # self.clip_to_image(masked_coords)
        x1, x2 = masked_coords[..., 1].min(), masked_coords[..., 1].max()
        y1, y2 = masked_coords[..., 0].min(), masked_coords[..., 0].max()

        x1 = torch.clamp(x1, 0, self.image_size[1])
        x2 = torch.clamp(x2, 0, self.image_size[1])
        y1 = torch.clamp(y1, 0, self.image_size[0])
        y2 = torch.clamp(y2, 0, self.image_size[0])

        return x1, y1, x2, y2

    def clip_to_image(self, coords):
        coords = coords.clone()
        def on_image(coord):
            return coord[0] >= 0 and coord[1] >= 0 and coord[0] <= self.image_size[1] and coord[1] <= self.image_size[0]

        coord_max = [self.image_size[1], self.image_size[1]]
        edges = [(0, 1), (4, 5), (0, 4), (1, 2), (5, 6), (1, 5), (2, 3), (6, 7), (2, 6), (3, 0), (7, 4), (3, 7), (0, 5), (1, 4)]

        for (i1, i2) in edges:
            start, end = coords[i1].float(), coords[i2].float()

            start_valid = on_image(start)
            end_valid = on_image(end)

            # dont have to interpret
            if start_valid and end_valid:
                continue
            
            # end -> start = (start - end)
            vec = (end - start)

            # 0 < start[0] + vec[0] * p < im_size[1]
            # -start[0] < vec[0] * p < im_size[1] - start[0]
            # -start[0] / vec[0] < p < (im_size[1] - start[0]) / vec[0]
            dim_mins = []
            dim_maxs = []
            for d in range(2):
                p1 = (-start[d] / vec[d]).clamp(0, 1)
                p2 = ((coord_max[d] - start[d]) / vec[d]).clamp(0, 1)

                dim_mins.append(min(p1, p2))
                dim_maxs.append(max(p1, p2))

            vmin = max(dim_mins)
            vmax = min(dim_maxs)

            new_start = start + vmin * vec
            new_end = start + vmax * vec

            coords[i1] = new_start.long()
            coords[i2] = new_end.long()

        return coords

    def get_clip_logits(self, images):
        image_features = self.clip.encode_image(images)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        # sims = text_features @ text_features.t()
        # sims.unsqueeze(0).unsqueeze(1)
        # save_image(sims, 'vith_sims.png')

        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

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
        cur_coords = cur_coords[..., [1, 0, 2]]

        # filter points outside of images
        # on_img = (
        #     (cur_coords[..., 1] < self.image_size[0])
        #     & (cur_coords[..., 1] >= 0)
        #     & (cur_coords[..., 0] < self.image_size[1])
        #     & (cur_coords[..., 0] >= 0)
        # )

        return cur_coords

    def forward(self, batch_dict, pred_dicts, keep_crops=False, relabel=True):
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

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        unif_grid = self.unif_grid

        if keep_crops:
            self.crop_infos = dict(crops=[], logits=[])

        for b in range(batch_size):
            cur_pred_dict = pred_dicts[b]

            cur_boxes = cur_pred_dict['pred_boxes']
            N = cur_boxes.shape[0]

            if N == 0:
                print('no boxes for clip!')
                print(cur_pred_dict)
                continue

            box_probs = torch.zeros((N, 6, 10), device='cuda', dtype=torch.half) # 6 cameras, 10 classes
            box_cam_mask = torch.zeros((N, 6), device='cuda')

            # box number
            cur_idx = torch.arange(0, N).reshape(N, 1).repeat(1, 8).reshape(N*8)
            corners = boxes_to_corners_3d(cur_boxes)
            corners = corners.reshape(-1, 3)

            cur_images = images[b]

            

            # filter points outside of images
            # on_img = (
            #     (cur_coords[..., 0] < self.image_size[0])
            #     & (cur_coords[..., 0] >= 0)
            #     & (cur_coords[..., 1] < self.image_size[1])
            #     & (cur_coords[..., 1] >= 0)
            # )

            for c in self.image_order:
                cur_coords = self.project_to_camera(batch_dict, corners, b, c)
                all_coords = cur_coords[0, ..., :2].long()
                all_depths = cur_coords[0, ..., 2]

                sampled_images = []
                sampled_idx = []

                last_idx = -1

                for i, idx in enumerate(cur_idx):
                    if idx == last_idx:
                        continue # repeated
                    last_idx = idx

                    coord_mask = (cur_idx == idx)
                    box_coords = all_coords[coord_mask]
                    box_depths = all_depths[coord_mask]

                    on_img = (
                        (box_coords[..., 1] < self.image_size[1])
                        & (box_coords[..., 1] >= 0)
                        & (box_coords[..., 0] < self.image_size[0])
                        & (box_coords[..., 0] >= 0)
                        & (box_depths >= 0.01) # in front of camera
                    )

                    if not on_img.any():
                        continue

                    # clip coords to image
                    x1, y1, x2, y2 = self.clip_coords(box_coords)

                    # this box occurs on this camera
                    box_cam_mask[idx, c] = 1

                    current_grid = unif_grid.clone()
                    current_grid = (current_grid - current_grid.min()) / (current_grid.max() - current_grid.min()) # 0, 1 grid

                    # CLIP performs better if the image is not squeezed too much
                    w, h = (x2 - x1), (y2 - y1)
                    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                    square_size = torch.maximum(w, h)

                    if square_size < self.min_crop_size:
                        continue

                    square_size = torch.clamp_min(square_size, self.min_crop_size)

                    # current_grid[..., 0] = current_grid[..., 0] * square_size + x1 #+ (w/2 - square_size/2)
                    # current_grid[..., 1] = current_grid[..., 1] * square_size + y1 #+ (h/2 - square_size/2)

                    current_grid[..., 0] = current_grid[..., 0] * square_size + x1
                    current_grid[..., 1] = current_grid[..., 1] * square_size + y1

                    # sample the image
                    grid_normalized = current_grid
                    grid_normalized[..., 0] = (grid_normalized[..., 0] / self.image_size[1]) * 2.0 - 1.0
                    grid_normalized[..., 1] = (grid_normalized[..., 1] / self.image_size[0]) * (2.0) - 1.0
                    
                    sampled_image = F.grid_sample(cur_images[[c]], grid=grid_normalized.cuda())


                    sampled_images.append(sampled_image)
                    sampled_idx.append(idx.item())

                if len(sampled_images) == 0:
                    continue

                sampled_images = torch.cat(sampled_images, dim=0)
                sampled_idx = torch.tensor(sampled_idx)


                # CLIP on crops on the current camera image
                with torch.no_grad():
                    logits_per_image, _ = self.get_clip_logits(sampled_images)
                    probs = logits_per_image.softmax(dim=-1)#.cpu()
                    # print('probs', probs.cpu().numpy())
                    box_probs[sampled_idx, c] = probs

                if keep_crops:
                    save_image(sampled_images, f'sampled_images_clip_{b}_{c}.png')

                    self.crop_infos['crops'].append(sampled_images.detach().cpu())
                    self.crop_infos['logits'].append(probs.detach().cpu())

            # mean over the camera images (that this box actually showed in)
            box_probs_mean = box_probs.sum(dim=1) / (1e-5 + box_cam_mask.sum(dim=-1).unsqueeze(1))

            # pred_labels = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)
            probs_max = torch.max(box_probs_mean.cpu(), dim=-1)
            pred_scores = probs_max.values
            pred_labels = probs_max.indices.flatten()

            pred_scores = torch.nan_to_num(pred_scores, nan=0.0)

            if keep_crops:
                self.crop_infos['crops'] = torch.cat(self.crop_infos['crops'], dim=0)
                self.crop_infos['logits'] = torch.cat(self.crop_infos['logits'], dim=0)
            
            if relabel:
                pred_dicts[b]['orig_labels'] = pred_dicts[b]['pred_labels'].clone() # 0 -> bg, 1 -> car, ...
                pred_dicts[b]['pred_labels'] = pred_labels + 1 # 0 -> bg, 1 -> car, ...
                pred_dicts[b]['pred_scores'] = pred_scores # replace with clip scores


        return pred_dicts