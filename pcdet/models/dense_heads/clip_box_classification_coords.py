import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from pcdet.utils import common_utils
from typing import Dict, List, Tuple, Union
from shapely.geometry import MultiPoint, box, LineString, Polygon

from PIL import Image
import clip
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

def clip_coords_hull(
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
            print('polygon', img_intersection)
            intersection_coords = np.array(
                [coord for coord in img_intersection.exterior.coords])
        else:
            print(f'img_intersection:{img_intersection} should be Polygon or LineString!')
            return None

        return intersection_coords
    else:
        return None

class CLIPBoxClassification(nn.Module):
    def __init__(self, image_size=[900, 1600], clip_model="ViT-L/14") -> None:
        super().__init__()

        self.image_order = [2, 0, 1, 5, 3, 4] # for plotting
        self.image_size = image_size

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

        model, preprocess = clip.load(clip_model, device='cuda')
        self.clip = model

        # ensembling text prompts
        self.text_features = self.forward_feature(self.all_class_names)

        self.min_crop_size = 64

        theta = torch.eye(2, 3).unsqueeze(0)
        self.unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]).cuda() # -1, 1 grid

    @torch.no_grad()
    def forward_feature(self, category_list):
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

    def forward(self, batch_dict, pred_dicts):
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

            cur_coords = corners
            cur_images = images[b]

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
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :].clone(), 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :].clone()

            # do image aug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            # filter points outside of images
            # on_img = (
            #     (cur_coords[..., 0] < self.image_size[0])
            #     & (cur_coords[..., 0] >= 0)
            #     & (cur_coords[..., 1] < self.image_size[1])
            #     & (cur_coords[..., 1] >= 0)
            # )

            for c in self.image_order:
                all_coords = cur_coords[c, :].long().cpu()

                sampled_images = []
                sampled_idx = []

                last_idx = -1

                for i, idx in enumerate(cur_idx):
                    if idx == last_idx:
                        continue # repeated
                    last_idx = idx

                    coord_mask = (cur_idx == idx)

                    coords2d = [[x[1], x[0]] for x in all_coords[coord_mask]]
                    box = post_process_coords(coords2d)

                    if box is None:
                        # box is not on the camera image
                        continue

                    box = torch.tensor(box, device='cuda')

                    x1, y1, x2, y2 = box

                    # this box occurs on this camera
                    box_cam_mask[idx, c] = 1

                    current_grid = unif_grid.clone()
                    current_grid = (current_grid - current_grid.min()) / (current_grid.max() - current_grid.min()) # 0, 1 grid

                    # CLIP performs better if the image is not squeezed too much
                    square_size = torch.maximum(x2 - x1, y2 - y1)
                    square_size = torch.clamp_min(square_size, self.min_crop_size)

                    current_grid[..., 0] = current_grid[..., 0] * square_size + x1 #+ curr_x_off
                    current_grid[..., 1] = current_grid[..., 1] * square_size + y1


                    # sample the image
                    grid_normalized = current_grid.clone()
                    grid_normalized[..., 0]  = (grid_normalized[..., 0] / self.image_size[1]) * 2.0 - 1.0
                    grid_normalized[..., 1] = (grid_normalized[..., 1] / self.image_size[0]) * 2.0 - 1.0
                    
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
                    box_probs[sampled_idx, c] = probs

            # mean over the camera images (that this box actually showed in)
            box_probs_mean = box_probs.sum(dim=1) / box_cam_mask.sum(dim=-1).unsqueeze(1)

            pred_labels = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)
            pred_labels = pred_labels.flatten().cpu()

            pred_dicts[b]['pred_labels'] = pred_labels + 1 # 0 -> bg, 1 -> car, ...

        return pred_dicts