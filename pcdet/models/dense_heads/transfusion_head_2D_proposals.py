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
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.utils import make_grid, save_image, draw_segmentation_masks
from torchvision.ops import batched_nms
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from ..preprocessed_detector import PreprocessedDetector


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    # points, is_numpy = check_numpy_to_torch(points)
    # angle, _ = check_numpy_to_torch(angle)

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
    return points_rot


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

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d

def get_2d_corners(xyxy):
    wh = xyxy[2:] - xyxy[0:2]

    template = xyxy.new_tensor((
        [1, 1], [1, -1], [-1, -1], [-1, 1]
    )) / 2

    box = template @ wh
    box += xyxy[0:2]

    return box

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

def draw_corners_on_image(corners, ax, color=(1, 1, 1), line_width=2, label='', max_num=500, tube_radius=None, linestyle='solid'):
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
    # x2, y1 = corners[:, 1].max(), corners[:, 0].min()
    # x2, y1 = corners[:, 0].max(), corners[:, 1].min()

    xt, yt = corners[:, 0].mean(), corners[:, 1].mean()

    if label != '':
        # ax.text(corners[6, 1] + 5, corners[6, 0] + 5, label, color=color)
        ax.text(xt, yt, label, color=(0, 0, 0))
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], color=color, linestyle=linestyle, linewidth=line_width)

        i, j = k + 4, (k + 1) % 4 + 4
        
        ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], color=color, linestyle=linestyle, linewidth=line_width)

        i, j = k, k + 4
        ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], color=color, linestyle=linestyle, linewidth=line_width)

    i, j = 0, 5
    ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], color=color, linestyle=linestyle, linewidth=line_width)

    i, j = 1, 4
    ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]], color=color, linestyle=linestyle, linewidth=line_width)

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

# with open('/home/uqdetche/OpenPCDet/tools/coco_classes_91.txt', 'r') as f:
#     lines = f.readlines()

#     coco_classes = ['background']
#     coco_classes.extend(lines)

#     coco_classes = [c.strip() for c in lines]


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

class TorchVisionDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()

    def train(self, mode): # NO TRAINING!!
        self.detector.train(False)
        self.detector.requires_grad_ = False

        for p in self.detector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images):
        images_list = list(image for image in images)
        predictions = self.detector(images_list)

        labels = []
        boxes = []
        scores = []
        idx = []
        for i, pred in enumerate(predictions):
            indices = batched_nms(pred['boxes'], pred['scores'], pred['labels'], iou_threshold=0.7)
            labels.append(pred['labels'][indices])
            boxes.append(pred['boxes'][indices])
            scores.append(pred['scores'][indices])
            idx.append(torch.zeros((pred['labels'][indices].shape[0])) + i)
        
        labels = torch.cat(labels, dim=0)
        boxes = torch.cat(boxes, dim=0)
        scores = torch.cat(scores, dim=0)
        idx = torch.cat(idx, dim=0)

        return boxes, labels, scores, idx

class ShapeGatherer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.num_classes = 10

    # def check_pt_in_box(self, box3d, pt):
    #     """Checks if pt is in the 2d bev box, and then if the z is within the z range of the box.

    #     Args:
    #         box3d (_type_): _description_
    #         pt (_type_): _description_
    #     """
    #     def tri_pt_area(pt1, pt2, pt3):
    #         return torch.abs( (pt2[0] * pt1[1] - pt1[0] * pt2[1]) + (pt3[0] * pt2[1] - pt2[0] * pt3[1]) + (pt1[0] * pt3[1] - pt3[0] * pt1[1]) ) / 2

    #     corners = boxes_to_corners_3d(box3d)

    #     # https://stackoverflow.com/questions/17136084/checking-if-a-point-is-inside-a-rotated-rectangle
    #     sum_of_areas = 

    def points_in_boxes(self, boxes3d, points):
        N = boxes3d.shape[0]

        points = points.clone()
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        # instead of rotating the box, rotate the pointcloud (so can simply threshold the points)
        points = points[:, None, :] - boxes3d[None, :, 0:3] # centre the points
        points = points.permute(1, 0, 2)

        points = rotate_points_along_z(points, - boxes3d[:, 6]).view(N, -1, 3) # rotate by negative angle

        in_box = (
            (points[..., 0] >= corners3d[..., 0].min(dim=1, keepdim=True).values) &
            (points[..., 0] <= corners3d[..., 0].max(dim=1, keepdim=True).values) &
            
            (points[..., 1] >= corners3d[..., 1].min(dim=1, keepdim=True).values) &
            (points[..., 1] <= corners3d[..., 1].max(dim=1, keepdim=True).values) &
            
            (points[..., 2] >= corners3d[..., 2].min(dim=1, keepdim=True).values) &
            (points[..., 2] <= corners3d[..., 2].max(dim=1, keepdim=True).values)
            )

        return in_box, points

    def forward(self, batch_dict):
        DEBUG = True

        # get points
        batch_size = batch_dict['batch_size']
        B = batch_size

        gt_boxes = batch_dict['gt_boxes']
        gt_bboxes_3d = gt_boxes[...,:-1]
        gt_labels_3d = gt_boxes[...,-1].long() - 1
        
        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            cur_gt_bboxes_3d = gt_bboxes_3d[b]
            valid_idx = []
            # filter empty boxes
            for i in range(len(cur_gt_bboxes_3d)):
                if cur_gt_bboxes_3d[i][3] > 0 and cur_gt_bboxes_3d[i][4] > 0:
                    valid_idx.append(i)

            # remove the empties
            cur_gt_bboxes_3d = cur_gt_bboxes_3d[valid_idx]
            cur_gt_labels = gt_labels_3d[b][valid_idx]

            in_boxes, box_relative_pts = self.points_in_boxes(cur_gt_bboxes_3d, cur_points)

                # get points in the 
                        # if DEBUG:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)
            
            N = cur_gt_bboxes_3d.shape[0]

            corners = boxes_to_corners_3d(cur_gt_bboxes_3d)
            corners = corners.reshape(N, 8, 3)

            # ax.scatter(pts_bev[..., 0], pts_bev[..., 1], c=pts_bev[..., 2])

            max_b_idx = 0
            max_in_box = 0

            all_car_relative_pts = []
            all_car_relative_pts_idx = []


            for b_idx, (box, label) in enumerate(zip(cur_gt_bboxes_3d, cur_gt_labels)):
            # for b_idx in range(N):
                color = PALETTE[b_idx % len(PALETTE)]

                points_in_box = cur_points[in_boxes[b_idx]].cpu()

                if label == 0:
                    cur_relative_pts = box_relative_pts[b_idx, in_boxes[b_idx]]
                    # box_half_size = cur_gt_bboxes_3d[max_b_idx, 3:6] / 2
                    # all_car_relative_pts.append(cur_relative_pts / box_half_size)
                    all_car_relative_pts.append(cur_relative_pts)
                    all_car_relative_pts_idx.extend([b_idx] * cur_relative_pts.shape[0])


                if points_in_box.shape[0] > max_in_box:
                    max_in_box = points_in_box.shape[0]
                    max_b_idx = b_idx

                # print('corners sub', corners[b_idx])
                draw_corners_on_image(corners[b_idx].cpu(), ax, color=color, label=f"{points_in_box.shape[0]}")
                ax.scatter(points_in_box[..., 0], points_in_box[..., 1], color=color)

            # ax.scatter(gt_boxes[..., 0], gt_boxes[..., 1], color=(0, 0, 1), label='gt boxes', s=100)
            # ax.scatter(query_pos[..., 0].cpu(), query_pos[..., 1].cpu(), color=(1, 0, 0), label='2d box ->3d queries')
            plt.legend()
            plt.savefig(f'points_in_boxes.png', bbox_inches='tight')
            plt.close()

            all_car_relative_pts = torch.cat(all_car_relative_pts, dim=0)

            # show the 3d shape of the most densely packed box
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # points_in_box = cur_points[in_boxes[max_b_idx]].cpu()
            # points_in_box = box_relative_pts[max_b_idx, in_boxes[max_b_idx]]
            # box_half_size = cur_gt_bboxes_3d[max_b_idx, 3:6] / 2 # distance from the centre

            # all_car_relative_pts
            # points_in_box /= box_half_size

            points_in_box = all_car_relative_pts.cpu()
    
            ax.scatter(points_in_box[..., 0], points_in_box[..., 1], points_in_box[..., 2], c=[PALETTE[i % len(PALETTE)] for i in all_car_relative_pts_idx])
            low, up = points_in_box.min(), points_in_box.max()
            ax.set_xlim((low, up))
            ax.set_ylim((low, up))
            ax.set_zlim((low, up))

            plt.legend()
            plt.savefig(f'most_packed.png', bbox_inches='tight')
            plt.close()


class TransFusionHead2DProposals(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """
    def __init__(
        self,
        model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True,
    ):
        super(TransFusionHead2DProposals, self).__init__()

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        self.num_classes = 10 # manual for one_hot

        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = class_names

        self.nusc_to_idx = {c:i for i, c in enumerate(self.all_class_names)}
        self.coco_to_idx = {c:i for i, c in enumerate(coco_classes)}

        # self.coco_to_nuscenes = {k: j for j, nusc_cls in enumerate(self.all_class_names) for k in range(len(coco_classes)) if nusc_cls == coco_classes[k]}
        
        self.nuscenes_to_coco_names = {
            'car': 'car',
            'truck': 'truck',
            'construction_vehicle': '',
            'bus': 'bus',
            'trailer': '',
            'barrier': '',
            'motorcycle': 'motorcycle',
            'bicycle': 'bicycle',            
            'pedestrian': 'person',
            'traffic_cone': ''
        }

        self.coco_to_nuscenes_idx = {self.coco_to_idx[coco_name]: self.nusc_to_idx[nusc_name] 
                                     for nusc_name, coco_name in self.nuscenes_to_coco_names.items() if coco_name != ''}

        # for k, v in self.coco_to_nuscenes_idx.items():
            # print(f'match {coco_classes[k]} with {self.all_class_names[v]}')

        self.known_class_idx = [i for i, cls in enumerate(self.all_class_names) if cls in self.known_class_names]
        print("known class idx", [(i, self.all_class_names[i]) for i in self.known_class_idx])

        self.image_order = [4, 2, 0, 1, 5, 3]


        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')

        self.image_size = self.model_cfg.IMAGE_SIZE
        # self.feature_size = self.model_cfg.FEATURE_SIZE

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

        # a shared convolution
        # self.shared_conv = nn.Conv2d(in_channels=input_channels,out_channels=hidden_channel,kernel_size=3,padding=1)

        # the dense heatmap (we do not want this)
        # layers = []
        # layers.append(BasicBlock2D(hidden_channel,hidden_channel, kernel_size=3,padding=1,bias=bias))
        # layers.append(nn.Conv2d(in_channels=hidden_channel,out_channels=num_class,kernel_size=3,padding=1))
        # self.heatmap_head = nn.Sequential(*layers)

        # self.image_detector = TorchVisionDetector() # only our wanted coco classes
        preds_path = "/home/uqdetche/GLIP/OWL_"
        camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

        self.image_detector = PreprocessedDetector([preds_path + f"{cam_name}.json" for cam_name in camera_names])
        self.shape_gatherer = ShapeGatherer()

        # self.frustum = self.create_frustum()
        # self.D = self.frustum.shape[0]

        text_dim = 256 

        self.max_proposals = 200
        self.depth_bins = 20

        self.depth_qs = torch.linspace(0.25, 0.75, self.depth_bins, device='cuda')

        # self.coords_norm = torch.tensor([np.max([abs(x) for x in self.point_cloud_range[[i, i+3]] for i in range(3)])])
        self.coords_norm = torch.tensor([max(abs(self.point_cloud_range[i]), abs(self.point_cloud_range[i + 3])) for i in range(3)], device='cuda')
        # print('coords norm', self.coords_norm)

        self.point_cloud_min = torch.tensor(self.point_cloud_range[0:3], device='cuda')
        self.point_cloud_max = torch.tensor(self.point_cloud_range[3:], device='cuda')

        self.box_coords_encoder = nn.Linear(8*3, hidden_channel, bias=True)
        self.query_pos_encoder = nn.Linear(3, hidden_channel, bias=True)
        # self.query_var_encoder = nn.Linear(3, hidden_channel, bias=False)
        self.query_depth_dist_encoder = nn.Linear(self.depth_bins, hidden_channel, bias=True)

        self.query_norm = nn.LayerNorm(hidden_channel)

        # self.query_text_encoding = nn.Conv1d(text_dim, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        # self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
        #         self_posembed=PositionEmbeddingLearned(2, hidden_channel),
        #         cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
            # )
        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        # heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        # self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride
        y_size = self.grid_size[1] // self.feature_map_stride
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        # print('bev pos', self.bev_pos.shape, 'frustum', self.frustum.shape)
        print('x_size', x_size, y_size)

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

    def create_frustum(self):
        iH, iW = self.image_size
        # fH, fW = self.feature_size
        fH, fW = iH, iW # want the frustum to be for each pixel (as we have bboxes, not features)

        ds = torch.arange(1.0, 60.0, 0.5, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)
        
        return nn.Parameter(frustum, requires_grad=False)

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

    def get_bbox_frustum_lidar(self, bboxes):
        for box in bboxes:
            print('get_bbox_frustum_lidar', box)

    def predict(self, batch_dict):
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
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]


        batch_size = batch_dict['batch_size']
        # inputs = batch_dict['spatial_features_2d']

        # query initialization
        images = batch_dict['camera_imgs']

        # query positions and labels
        query_pos_batch = torch.zeros((batch_size, self.max_proposals, 3), device='cuda')
        query_var_batch = torch.zeros((batch_size, self.max_proposals, 3), device='cuda')
        query_box_batch = torch.zeros((batch_size, self.max_proposals, 8*3), device='cuda')
        query_depth_batch = torch.zeros((batch_size, self.max_proposals, self.depth_bins), device='cuda')

        query_score_batch = torch.zeros((batch_size, self.max_proposals, 1), device='cuda')
        query_labels_batch = torch.zeros((batch_size, self.max_proposals), dtype=torch.long, device='cuda')

        query_cam_idx = torch.zeros((batch_size, self.max_proposals), dtype=torch.long, device='cuda')
        query_batch_idx = torch.zeros((batch_size, self.max_proposals), dtype=torch.long, device='cuda')

        # camera_query_cam_idx = []
        # camera_query_batch_idx = []
        # camera_query_color = []

        DEBUG = False

        # depth = torch.zeros(batch_size, images.shape[1], 1, *self.image_size).to(inputs.device)
        # depth_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)

        for b in range(batch_size):
            detector_batch_mask = (det_batch_idx == b)
            # print('detector batch mask', detector_batch_mask)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            if cur_boxes.shape[0] >= self.max_proposals:
                print('there are more detected boxes than max proposals!', self.max_proposals)

            # sort by highest score (as we have a maximum number of proposals )
            indices = torch.sort(cur_scores, descending=True).indices
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = cur_boxes[indices], cur_labels[indices], cur_scores[indices], cur_cam_idx[indices]

            if DEBUG and b == 0:
                # all in one image
                images_joined = make_grid(images[b, self.image_order], nrow=6, padding=0, normalize=True, scale_each=True)
                # save_image(images_joined, '2djoined_.png')

                wj, hj = images_joined.shape[-2:]
                images_joined_np = images_joined.cpu().permute(1, 2, 0).numpy()

                dpi = 100
                fig = plt.figure(frameon=False, dpi=dpi)
                fig.set_size_inches(wj/50, hj/50)
                ax = fig.gca()
                ax.imshow(images_joined_np)
                plt.axis('off')
                curr_x_off = 0

            proj_points, proj_points_cam_mask = self.project_to_image(batch_dict, batch_idx=b)

            batch_box_idx = 0

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]


                cam_points = proj_points[c, proj_points_cam_mask[c]]
                assert cam_points.numel() > 0, "no points on this view!"

                if batch_box_idx > self.max_proposals:
                    break

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < 0.1:
                        continue

                    if isinstance(self.image_detector, TorchVisionDetector) and label.item() not in self.coco_to_nuscenes_idx.keys():
                        continue

                    if batch_box_idx >= self.max_proposals:
                        print('too many 2d detections!', batch_box_idx, self.max_proposals, cur_boxes.shape)
                        break

                    x1, y1, x2, y2 = box.cpu()


                    on_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    box_points = cam_points[on_box]

                    if box_points.numel() == 0:
                        continue

                    # box frustum
                    frustum_min = torch.quantile(box_points[:, 2], 0.25)
                    frustum_max = torch.quantile(box_points[:, 2], 0.75)

                    box = box.cuda()
                    xyzxyz = torch.cat([box[0][None], box[1][None], frustum_min[None], box[2][None], box[3][None], frustum_max[None]])

                    frust_box = get_cam_frustum(xyzxyz)
                    frust_box = self.get_geometry_at_image_coords(frust_box, [c] * 8, [b] * 8, # 8 corners in a box
                        camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
                        post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                    )

                    # normalize frust_box by point cloud range
                    # frust_box = frust_box / self.coords_norm
                    frust_box = (frust_box - self.point_cloud_min) / (self.point_cloud_max - self.point_cloud_min)

                    query_box_batch[b, batch_box_idx] = frust_box.reshape(-1)

                    # depth distribution
                    depth_quantiles = torch.quantile(box_points[:, 2], q=self.depth_qs)
                    query_depth_batch[b, batch_box_idx] = depth_quantiles / self.point_cloud_max.max()

                    # get the weighted depth based on the distance to the centre (in pixels)
                    box_centre = torch.cat([(x1[None] + x2[None]) / 2, (y1[None] + y2[None]) / 2]).reshape(1, -1).cuda()
                    dists = torch.cdist(box_centre, box_points[:, 0:2])
                    dists_ranking = torch.softmax(- dists, dim=-1) # softmin

                    weighted_centre = dists_ranking @ box_points

                    # weighted centre
                    # camera_query_pos.append(weighted_centre)
                    query_pos_batch[b, batch_box_idx] = weighted_centre
                    # query_var_batch[b, batch_box_idx] = torch.var(box_points, dim=0)
                    if isinstance(self.image_detector, TorchVisionDetector):
                        query_labels_batch[b, batch_box_idx] = self.coco_to_nuscenes_idx[label.item()]
                    else:
                        assert label >= 0 and label < self.num_classes
                        query_labels_batch[b, batch_box_idx] = label

                    query_score_batch[b, batch_box_idx] = score
                    
                    # camera and batch indices
                    query_batch_idx[b, batch_box_idx] = b
                    query_cam_idx[b, batch_box_idx] = c

                    batch_box_idx += 1
                    

                    if DEBUG and b == 0:
                        label_txt = f"{self.all_class_names[label.item()]} {box_points.numel()}, {score:.2f}"

                        color = PALETTE[label.item() % len(PALETTE)]
                        # camera_query_color.append(color)
                        # ax.scatter(box_points[..., 0].cpu() + curr_x_off, box_points[..., 1].cpu(), color=color, s=5)
                        ax.text(x2 + 5 + curr_x_off, y1 + 5, label_txt, color=color)
                        ax.add_patch(Rectangle(xy=[x1+curr_x_off, y1], width=x2-x1, height=y2-y1, fill=False, color=color))
                        
                if DEBUG and b == 0:
                    curr_x_off += self.image_size[1]

            if DEBUG and b == 0:
                plt.savefig(f'images_joined_2dhead_preproc.png', bbox_inches='tight', dpi=dpi)
                plt.close()

        # centre points
        query_pos = self.get_geometry_at_image_coords(query_pos_batch.reshape(-1, 3), query_cam_idx.reshape(-1), query_batch_idx.reshape(-1),
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )

        query_pos = query_pos.reshape(-1, self.max_proposals, 3)
        # print('query pos', query_pos.shape)

        # print('query_box_batch', query_box_batch.min(), query_box_batch.max())
        # print('query_depth_batch', query_depth_batch.min(), query_depth_batch.max())

        query_feat = self.query_pos_encoder(query_pos)
        query_feat += self.box_coords_encoder(query_box_batch)
        query_feat += self.query_depth_dist_encoder(query_depth_batch)
        # query_feat += self.query_var_encoder(query_var_batch)

        query_feat = self.query_norm(query_feat)
        
        query_feat = query_feat.permute(0, 2, 1)
        res_layer = self.prediction_head(query_feat)


        self.query_labels = query_labels_batch
        # class output (purely based on )
        one_hot = F.one_hot(query_labels_batch, num_classes=self.num_classes).permute(0, 2, 1)
        # zero for no object
        one_hot = one_hot * query_score_batch.permute(0, 2, 1)
        res_layer['heatmap'] = one_hot

        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)[:, :2]
        # print('center', res_layer['center'].shape, self.coords_norm.shape)qq
        # res_layer["center"] = res_layer["center"] * self.coords_norm[:2].reshape(1, -1, 1)
        # print(res_layer['center']) 
        res_layer["query_heatmap_score"] = one_hot

    
        if DEBUG:
            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)

            pts_bev = batch_dict['points'][..., 1:4].cpu()
            # gt_boxes = batch_dict['gt_boxes'][..., 0:3].cpu()
            
            B, N = batch_dict['gt_boxes'].shape[0:2]

            gt_boxes = batch_dict['gt_boxes'].reshape(B*N, -1)
            # gt_boxes = gt_boxes[gt_boxes[..., 0] == 0] # batch 0

            # B, N = 1, gt_boxes.shape[0]

            corners = boxes_to_corners_3d(gt_boxes)
            corners = corners.reshape(B*N, 8, 3)

            ax.scatter(pts_bev[..., 0], pts_bev[..., 1], color=(0, 0, 0))

            for b_idx in range(B*N):
                # print('corners sub', corners[b_idx])
                draw_corners_on_image(corners[b_idx].cpu(), ax, color=(0, 0, 1))

            # ax.scatter(gt_boxes[..., 0], gt_boxes[..., 1], color=(0, 0, 1), label='gt boxes', s=100)
            ax.scatter(query_pos[..., 0].cpu(), query_pos[..., 1].cpu(), color=(1, 0, 0), label='2d box ->3d queries')
            ax.scatter(res_layer["center"][..., 0].detach().cpu(), res_layer["center"][..., 1].detach().cpu(), color=(0, 1, 0), label='predicted centres')
            plt.legend()
            plt.savefig(f'bev_2dhead_queries_preproc.png', bbox_inches='tight', dpi=dpi)
            plt.close()

        return res_layer

    def project_to_image(self, batch_dict, batch_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        batch_size = batch_dict['batch_size']

        points = batch_dict['points']
        # print('points shape', points.shape)
        points_idx = points[..., 0]

        batch_mask = (points_idx == batch_idx)
        # print('batch mask', batch_mask.shape)
        points = points[batch_mask, 1:4]

        cur_coords = points.clone()

        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
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

    def forward(self, batch_dict):
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
        # matched_ious = np.mean(res_tuple[5])
        matched_ious = torch.cat(res_tuple[5], dim=0)
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
        # for idx in range(len(gt_bboxes_3d)):
        #     width = gt_bboxes_3d[idx][3]
        #     length = gt_bboxes_3d[idx][4]
        #     width = width / self.voxel_size[0] / self.feature_map_stride
        #     length = length / self.voxel_size[1] / self.feature_map_stride
        #     if width > 0 and length > 0:
        #         radius = centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[0]
        #         radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
        #         x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

        #         coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
        #         coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

        #         center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
        #         center_int = center.to(torch.int32)
        #         centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)


        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), ious[None], heatmap[None])
        # return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        # loss_heatmap = self.loss_heatmap(
        #     clip_sigmoid(pred_dicts["dense_heatmap"]),
        #     heatmap,
        # ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        # loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        # loss_all += loss_heatmap * self.loss_heatmap_weight

        matched_ious = matched_ious.reshape(-1)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)
        cls_score_sigmoid = cls_score.clone().detach().sigmoid()

        pos_labels_mask = labels < self.num_classes
        pos_labels = labels[labels < self.num_classes]
        matched_cls_score_sigmoid = cls_score_sigmoid[pos_labels_mask]

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

        # loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = matched_ious[labels < self.num_classes].mean()
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
