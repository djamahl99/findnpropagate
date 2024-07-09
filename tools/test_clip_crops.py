import argparse
import glob
from pathlib import Path


import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
from torchvision.transforms import ToPILImage

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, NuScenesDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from typing import Dict

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from pcdet.models.backbones_image import MaskCLIP
from PIL import Image
import clip
from clip import available_models, tokenize
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from prompts import imagenet_templates

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208)]

PALETTE = [[x/255 for x in y] for y in PALETTE]

all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
all_colors = ['red', 'red', 'red', 'red', ]
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def draw_corners_on_image(corners, ax, color=(1, 1, 1), line_width=2, label='', max_num=500, tube_radius=None):
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

    x2, y1 = corners[:, 1].max(), corners[:, 0].min()

    if label != '':
        # ax.text(corners[6, 1] + 5, corners[6, 0] + 5, label, color=color)
        ax.text(x2 + 5, y1 + 5, label, color=color)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color)

        i, j = k + 4, (k + 1) % 4 + 4
        
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color)

        i, j = k, k + 4
        ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color)


    i, j = 0, 5
    ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color)

    i, j = 1, 4
    ax.plot([corners[i, 1], corners[j, 1]], [corners[i, 0], corners[j, 0]], color=color)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/box_proj.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

# dict_keys(['points', 'frame_id', 'metadata', 'gt_boxes', 'image_paths', 'lidar2camera', 
#   'lidar2image', 'camera2ego', 'camera_intrinsics', 'camera2lidar', 'camera_imgs', 'ori_shape', 
#       'img_process_infos', 'lidar_aug_matrix', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'img_aug_matrix', 'batch_size'])

class ImageBoxSampling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [448, 800]

        self.num_sample_pts = 100

        # self.maskclip = MaskCLIP(dict()).to('cuda')

        model, preprocess = clip.load("ViT-L/14", device='cuda')
        self.clip = model

        # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        # self.text = clip.tokenize([f"a {c}" for c in all_class_names]).to('cuda')

        self.text_features = self.forward_feature(all_class_names)

        theta = torch.eye(2, 3).unsqueeze(0)
        self.unif_grid = F.affine_grid(theta=theta, size=[1, 3, 224, 224]) # -1, 1 grid

    @torch.no_grad()
    def forward_feature(self, category_list):
        templates = imagenet_templates
        texts = [template.format(cetegory) for cetegory in category_list for template in templates] #format with class
        print('texts', len(texts))
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
        print('text_features', self.text_features)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)

        sims = text_features @ text_features.t()
        sims.unsqueeze(0).unsqueeze(1)
        save_image(sims, 'vith_sims.png')

        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

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

        B, N = batch_dict['gt_boxes'].shape[0:2]

        if B*N == 0: # if there are no boxes
            return {
                'pred_labels': None,
                'true_labels': None,
                'acc': 0,
            }

        gt_boxes = batch_dict['gt_boxes'].reshape(B*N, -1)
        corners = boxes_to_corners_3d(gt_boxes)
        corners = corners.reshape(B, N*8, 3)

        gt_labels = batch_dict['gt_boxes'][..., -1].clone().long()

        # box number
        box_idx = torch.arange(0, B*N).reshape(B, N, 1).repeat(1, 1, 8).reshape(B, N*8)
        coord_labels = gt_labels.reshape(B, N, 1).repeat(1, 1, 8).reshape(B, N*8)

        unif_grid = self.unif_grid

        box_probs = torch.zeros((B*N, 6, 10), device='cuda', dtype=torch.half) # 6 cameras, 10 classes
        box_cam_mask = torch.zeros((B*N, 6), device='cuda')

        pred_labels = []
        true_labels = gt_labels.reshape(-1) - 1 # 1 indexed

        for b in range(batch_size):
            cur_coords = corners[b, :, :3]
            cur_idx = box_idx[b]
            cur_labels = coord_labels[b]
            cur_images = images[b]

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

            for c in self.image_order:
                all_coords = cur_coords[c, :].long().cpu()

                # get boxes with at least one corner on the current image
                masked_idx = cur_idx[on_img[c]]
                masked_labels = cur_labels[on_img[c]]

                sampled_images = []
                sampled_idx = []

                last_sample_idx = -1

                for i, idx in enumerate(masked_idx):
                    if idx == last_sample_idx:
                        # avoid resampling due to multiple points per box
                        continue
                    last_sample_idx = idx
                    coord_mask = (cur_idx == idx)
                    box_coords = all_coords[coord_mask].clone().cpu().numpy()

                    box_cam_mask[idx, c] = 1 # this box occurs on this camera

                    lbl_idx = masked_labels[i]
                    # true_labels.append(lbl_idx.item() - 1)

                    x1, x2 = all_coords[coord_mask, 1].min(), all_coords[coord_mask, 1].max()
                    y1, y2 = all_coords[coord_mask, 0].min(), all_coords[coord_mask, 0].max()

                    x1 = torch.clamp(x1, 0, self.image_size[1])
                    x2 = torch.clamp(x2, 0, self.image_size[1])
                    y1 = torch.clamp(y1, 0, self.image_size[0])
                    y2 = torch.clamp(y2, 0, self.image_size[0])

                    current_grid = unif_grid.clone()
                    current_grid = (current_grid - current_grid.min()) / (current_grid.max() - current_grid.min()) # 0, 1 grid

                    square_size = torch.maximum(x2 - x1, y2 - y1)
                    min_size = 64
                    square_size = torch.clamp_min(square_size, min_size)

                    # x 
                    # current_grid[..., 0] = current_grid[..., 0] * (x2 - x1) + x1 #+ curr_x_off
                    # current_grid[..., 1] = current_grid[..., 1] * (y2 - y1) + y1

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
                # save_image(sampled_images, 'sampled_images.png')

                # CLIP on crops on the current camera image
                with torch.no_grad():
                    # logits_per_image, logits_per_text = self.clip(sampled_images, self.text)
                    logits_per_image, _ = self.get_clip_logits(sampled_images)
                    probs = logits_per_image.softmax(dim=-1)#.cpu()
                    box_probs[sampled_idx, c] = probs

        # mean over the camera images (that this box actually showed in)
        box_probs_mean = box_probs.sum(dim=1) / box_cam_mask.sum(dim=-1).unsqueeze(1)

        argmax = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)

        pred_labels = argmax.flatten().cpu()
        true_labels = true_labels.cpu()

        acc_fn = Accuracy(task='multiclass', num_classes=10)
        batch_acc = acc_fn(pred_labels, true_labels)

        return {
            'pred_labels': pred_labels.cpu(),
            'true_labels': true_labels.cpu(),
            'acc': batch_acc
        }


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )
    # ROOT_DIR = Path('')
    demo_dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=all_class_names,
            # root_path=ROOT_DIR / 'data' / 'nuscenes',
            root_path=Path('/home/uqdetche/OpenPCDet/data/nuscenes'),
            logger=logger, training=False
        )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    box_sampling = ImageBoxSampling()

    pred_labels = None
    true_labels = None
    acc_fn = MulticlassAccuracy(num_classes=10, average=None)


    with torch.no_grad():
        # for idx, data_dict in tqdm(enumerate(demo_dataset)):
        for data_dict in tqdm(demo_dataset):
            # logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            pred_dict = box_sampling(data_dict)

            if pred_dict['true_labels'] is None: # this batch has boxes
                continue

            if pred_labels is None and true_labels is None:
                pred_labels = pred_dict['pred_labels']
                true_labels = pred_dict['true_labels']
            else:
                pred_labels = torch.cat((pred_labels, pred_dict['pred_labels']))
                true_labels = torch.cat((true_labels, pred_dict['true_labels']))

            # full_accs = acc_fn(pred_labels, true_labels)
            batch_accs = acc_fn(pred_dict['pred_labels'], pred_dict['true_labels'])

            # print("full_accs", full_accs)

            # for cls_, f_acc, b_acc in zip(all_class_names, full_accs, batch_accs):
                # print(f"{cls_} ({b_acc*100:.2f}, {f_acc*100:.2f})% Accuracy (batch, full)")

            for cls_, b_acc in zip(all_class_names, batch_accs):
                print(f"{cls_} {b_acc*100:.2f}% Accuracy (batch)")

    torch.save({'pred_labels': pred_labels, 'true_labels': true_labels}, 'acc_dict.pth')

            # exit()
    full_accs = acc_fn(pred_labels, true_labels)

    for cls_, f_acc in zip(all_class_names, full_accs):
        print(f"{cls_} {f_acc*100:.2f}% Accuracy")

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
