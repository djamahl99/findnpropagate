from typing import Any
import torch
from torch import Tensor
import numpy as np

from typing import Dict, List
import os

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils.common_utils import rotate_points_along_z

def valid_boxes(cur_gt_bboxes_3d: Tensor) -> list:
    valid_idx = []
    # filter empty boxes
    for i in range(len(cur_gt_bboxes_3d)):
        if cur_gt_bboxes_3d[i][3] > 0 and cur_gt_bboxes_3d[i][4] > 0: # size > 0
            valid_idx.append(i)

    return cur_gt_bboxes_3d[valid_idx]

# class ObjectSample(object):
#     num_points: int
#     points: Tensor
#     box: Tensor
#     label: int
#     l: float
#     w: float
#     h: float

#     def __init__(self, points: Tensor, box: Tensor) -> None:
#         self.num_points = points.shape[0]
#         self.label = box[..., -1].item() 
#         box = box.reshape(1, 8)

#         self.l, self.w, self.h = [x.item() for x in box[0, [3, 4, 5]]]

#         points, box = self.transform_to_axis_aligned(points, box)
#         self.points, self.box = points, box

#         self.points = points.cpu()
#         self.box = box
    
#     def transform_to_axis_aligned(self, points: Tensor, box3d: Tensor) -> Tensor:
#         """Translates and then rotates points to the box coordinate frame.

#         Args:
#             points (Tensor): (N, 3) points
#             box3d (Tensor): (1, 7 + C) box

#         Returns:
#             Tensor: (N, 3) translated + rotated points
#         """
#         points = points.clone()
#         points = points[:, :3] - box3d[:, 0:3] # centre the points
#         # points = points.permute(1, 0, 2)

#         points = rotate_points_along_z(points[None], - box3d[:, 6]).view(-1, 3) # rotate by negative angle

#         # now centre box
#         box3d[:, 0:3] = 0
#         # remove rotation
#         box3d[:, 6] = 0

#         return points, box3d
    
#     def get_sample_points(self, sample_box: Tensor) -> Tensor:
#         points = self.points.clone()

#         points = rotate_points_along_z(points[None], sample_box[:, 6]).view(-1, 3) # rotate
#         points = points[:, :] + sample_box[:, 0:3] # translate

#         return points

#     def sample(self, gt_boxes: Tensor, pseudo_boxes: Tensor, max_iou=0.1) -> Tensor:
#         """ Sample from this object
#             - adds random x, y and then infers z from gt? and makes sure that we are not overlapping with other boxes
#         """
#         obj_bottoms = (gt_boxes[..., 2] - gt_boxes[..., 5] / 2).cpu()
#         bot_min, bot_max = obj_bottoms.min().item(), obj_bottoms.max().item()
#         bev_min, bev_range = -50.0, 100.0

#         valid = False
        
#         while not valid:
#             rx, ry, rz = np.random.rand((3))

#             x = bev_min + rx * bev_range
#             y = bev_min + ry * bev_range
#             z = bot_min + rz * (bot_max - bot_min) + self.h / 2

#             alpha = np.pi * np.random.rand()

#             sampled_box = gt_boxes.new_tensor([x, y, z, self.l, self.w, self.h, alpha, self.label]).reshape(1, 8)

#             # check not overlapping with gt
#             ious = iou3d_nms_utils.boxes_iou_bev(sampled_box[:, :7], gt_boxes[:, :7])

#             if ious.numel() > 0 and ious.max() > max_iou:
#                 continue

#             # check not overlapping with pseudo
#             ious = iou3d_nms_utils.boxes_iou_bev(sampled_box[:, :7], pseudo_boxes[:, :7].cuda())

#             if ious.numel() == 0 or ious.max() < max_iou:
#                 valid = True

#         return sampled_box, self.get_sample_points(sampled_box)

# class PseudoSampler(object):
#     unknown_queue: Dict[int, List[ObjectSample]]
#     unknown_class_labels: list
#     max_samples_per_class: int
#     prop_per_unk: Dict[int, float] # includes all class
#     mom: float = 0.7
#     num_classes: int # unkn + known

#     def __init__(self, unknown_class_labels: list, max_samples_per_class: int = 100, num_classes: int = 10) -> None:
#         self.unknown_class_labels = unknown_class_labels
#         self.max_samples_per_class = max_samples_per_class

#         # initialize queue
#         self.unknown_queue = {l: [] for l in self.unknown_class_labels}
#         self.num_classes = num_classes

#         # uniform init
#         self.prop_per_unk = {idx: 1.0/float(len(self.unknown_class_labels)) for idx in self.unknown_class_labels}

#     def calc_seen_per_class(self, pseudo_boxes: Tensor, gt_boxes: Tensor) -> None:
#         pseudo_labels = pseudo_boxes[..., -1].clone().reshape(-1).long()

#         boxes_curr = (pseudo_labels > 0).sum().item()
#         boxes_curr = max(boxes_curr, 1e-7)

#         for idx in self.unknown_class_labels:
#             num_curr = (pseudo_labels == idx).sum().item()

#             # EMA
#             self.prop_per_unk[idx] = self.prop_per_unk[idx] * self.mom + (num_curr / boxes_curr) * (1 - self.mom)

#     def __call__(self, batch_dict, pseudo_boxes: Tensor, gt_boxes: Tensor, num_proposals: int=100) -> Dict:
#         self.calc_seen_per_class(pseudo_boxes, gt_boxes)

#         # number of samples we have collected
#         samples_per_label = {l: len(x) for l, x in self.unknown_queue.items()}

#         # pseudos_out = pseudo_boxes.new_zeros(batch_dict['batch_size'], 200, 8)
#         # pseudos_out = torch.zeros_like(pseudo_boxes)
#         num_proposals = max(num_proposals, pseudo_boxes.shape[1])
#         pseudos_out = gt_boxes.new_zeros(batch_dict['batch_size'], num_proposals, 8)

#         batch_points = []
        
#         for b in range(batch_dict['batch_size']):
#             cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:]
#             cur_pt_idx = batch_dict['points'][..., 0]

#             cur_gt = valid_boxes(gt_boxes[b])
#             cur_pseudo = valid_boxes(pseudo_boxes[b])

#             if cur_pseudo.numel() == 0:
#                 continue

#             # find pseudos that don't overlap with GT
#             ious = iou3d_nms_utils.boxes_iou3d_gpu(cur_pseudo[..., :7].cuda(), cur_gt[..., :7].cuda())
#             ious = ious.max(dim=1).values
#             ious_mask = (ious < 0.1)
            
#             cur_pseudo = cur_pseudo[ious_mask].contiguous()
#             pseudo_pt_indices = roiaware_pool3d_utils.points_in_boxes_gpu(cur_points[None, :, :3].cuda(), cur_pseudo[None, :, :7].cuda())

#             # remove pseudos with no points
#             valid_idx = []

#             num_pseudos = cur_pseudo.shape[0]

#             # collect object samples
#             for idx, box in enumerate(cur_pseudo):
#                 pt_mask = (pseudo_pt_indices[0] == idx)
#                 pseudo_points = cur_points[pt_mask]

#                 if pseudo_points.numel() == 0:
#                     continue

#                 valid_idx.append(idx)
                
#                 label = box[..., -1].item()

#                 if samples_per_label[label] >= self.max_samples_per_class:
#                     # find worst sample (e.g. num points)
#                     rpl_idx = np.argmin([x.num_points for x in self.unknown_queue[label]])
#                     self.unknown_queue[label][rpl_idx] = ObjectSample(pseudo_points.cpu(), box)
#                     # print('samples overloaded')
#                     # rpl_idx = min([(i, x) for i, x in enumerate(self.unknown_queue[label])], key=lambda _, obj_sample: obj_sample.num_points)
#                     # print('replace idx', rpl_idx)
#                 else:
#                     self.unknown_queue[label].append(ObjectSample(pseudo_points.cpu(), box))

#             # remove those with no points
#             pseudos_out[b, :len(valid_idx)] = cur_pseudo[valid_idx]

#             num_pseudos = len(valid_idx)

#             # number of objects for gt and pseudo
#             num_gt_and_pseudo = cur_gt.shape[0] + len(valid_idx)
#             num_samples = num_proposals - num_gt_and_pseudo # TODO: tune

#             print('pseudos/samples', num_pseudos, num_samples)

#             if num_samples <= 0:
#                 continue

#             # neg proportions
#             p = [1.0 - x for x in self.prop_per_unk.values()]
#             p_sum = sum(p)
#             p = [x/max(p_sum, 1e-7) for x in p]

#             labels_to_sample = np.random.choice(self.unknown_class_labels, size=num_samples, replace=True, p=p)
#             sample_idx = len(valid_idx)

#             # add some samples
#             for lbl in labels_to_sample:
#                 if samples_per_label[lbl] == 0:
#                     continue

#                 idx = np.random.choice(len(self.unknown_queue[lbl]))

#                 sample_box, sample_points = self.unknown_queue[lbl][idx].sample(cur_gt, cur_pseudo)
#                 pseudos_out[b, [sample_idx]] = sample_box


#                 sample_idx += 1

#         return pseudos_out

class PseudoLoader(object):
    """
    Loads pseudo labels for unknown classes in current batch and combines them with gt knowns.
    """
    def __init__(self, known_class_names: list, pseudo_path='pseudo_labels/frustum_proposals/'):
        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = known_class_names
        self.num_classes = len(self.all_class_names)

        self.is_known = {i: (cls in self.known_class_names) for i, cls in enumerate(self.all_class_names)}

        # in open vocab, training has less classes
        self.training = len(self.known_class_names) != self.num_classes

        # gt_labels are 1 indexed
        self.gt_known_to_full_labels = {(i + 1): (j + 1) for i, known_name in enumerate(self.known_class_names) for j, all_name in enumerate(self.all_class_names) if known_name == all_name}
        self.full_labels_to_gt_known = {v: k for k, v in self.gt_known_to_full_labels.items()}
        self.unknown_labels = [(i + 1) for i, cls_name in enumerate(self.all_class_names) if cls_name not in self.known_class_names]
        
        if self.training: # print unknown information (in testing we have the gt for all boxes)
            for i, cls in enumerate(self.all_class_names):
                mp_str = ""
                if (i + 1) in self.full_labels_to_gt_known:
                    mp_str = f"no. {self.full_labels_to_gt_known[i+1]} => {i+1} full labels"
                    mp_str += f", {self.all_class_names[i]} => {self.known_class_names[self.full_labels_to_gt_known[i + 1] - 1]}"
                print(f"{cls}: {'Known' if self.is_known[i] else 'Unknown'}", mp_str)

            print('unknown labels', self.unknown_labels)

        self.pseudos_missing = set()

        self.sampler = PseudoSampler(unknown_class_labels=self.unknown_labels, max_samples_per_class=100)

        self.pseudo_folder = pseudo_path

    def relabel_gt_boxes(self, gt_boxes: Tensor) -> Tensor:
        """
        As the idx changes when remove classes from the class list, need to 
        remap back to original idx for original 10 classes.

        Args:
            gt_boxes (Tensor): gt boxes

        Returns:
            Tensor: gt_boxes relabeled to full class idx
        """
        for b in range(gt_boxes.shape[0]):
            for i in range(gt_boxes.shape[1]):
                curr_label = int(gt_boxes[b, i, -1].item())
                if curr_label in self.gt_known_to_full_labels:
                    # only modify if not the non-match id
                    gt_boxes[b, i, -1] = self.gt_known_to_full_labels[curr_label]

        return gt_boxes

    def load_pseudos(self, batch_dict, unknowns_only=True) -> Tensor:
        """Load pseudo labels

        Args:
            batch_dict (dict): batch data
            unknowns_only (bool): only return pseudo boxes for unknowns
            
        Returns:
            Tensor: pseudo boxes (B, M, 8)
        """
        frame_ids = batch_dict['frame_id']

        batch_size = batch_dict['batch_size']

        batch_boxes = []
        batch_scores = []
        batch_labels = []

        max_num_boxes = 0

        for b, frame_id in enumerate(frame_ids):
            pseudo_path = self.pseudo_folder + f"{frame_id.replace('.', '_')}.pth"

            if not os.path.exists(pseudo_path):
                self.pseudos_missing.add(pseudo_path)
                print('psuedos dont exist!', self.pseudos_missing)
                batch_boxes.append(torch.zeros((0, 7)))
                batch_scores.append(torch.zeros((0)))
                batch_labels.append(torch.zeros((0), dtype=torch.long))

                continue

            preds = torch.load(pseudo_path, map_location='cpu')

            for preds_dict in preds: # should be just the one (batch-size 1)
                pred_boxes = preds_dict['pred_boxes']
                pred_scores = preds_dict['pred_scores']
                pred_labels = preds_dict['pred_labels']

                if unknowns_only: # don't need pseudo labels for knowns
                    pred_mask = torch.zeros_like(pred_labels, dtype=torch.bool)
                    for i, lbl in enumerate(pred_labels):
                        lbl_idx = lbl.item() - 1

                        pred_mask[i] = not self.is_known[lbl_idx]

                    # filter out known boxes
                    pred_boxes = pred_boxes[pred_mask]
                    pred_scores = pred_scores[pred_mask]
                    pred_labels = pred_labels[pred_mask]

                if pred_labels.shape[0] > max_num_boxes:
                    max_num_boxes = pred_labels.shape[0]

                batch_boxes.append(pred_boxes)
                batch_scores.append(pred_scores)
                batch_labels.append(pred_labels)

        # now turn into (B, N, 8) gt_boxes (needed to first calculate max number of boxes for padding)
        gt_boxes = torch.zeros((batch_size, max_num_boxes, 8), dtype=torch.float)

        for b, (pred_boxes, pred_scores, pred_labels) in enumerate(zip(batch_boxes, batch_scores, batch_labels)):
            gt_boxes[b, :pred_boxes.shape[0], :7] = pred_boxes
            gt_boxes[b, :pred_boxes.shape[0], -1] = pred_labels

        return gt_boxes
    
    def combine_gt_with_pseudos(self, gt_boxes: Tensor, pseudo_boxes: Tensor) -> Tensor:
        """Combine gt boxes for known classes with pseudo boxes for unknowns

        Args:
            gt_boxes (Tensor): gt boxes (B, N, 8) from dataloader, relabeled by relabel_gt_boxes
            pseudo_boxes (Tensor): pseudo boxes (B, M, 8)

        Returns:
            Tensor: combined boxes (B, N+M, 8)
        """

        MN = 0
        MN_ = gt_boxes.shape[1] + pseudo_boxes.shape[1] # adds redundant padding
        code_size = gt_boxes.shape[-1]
        ret_boxes = torch.zeros((gt_boxes.shape[0], MN_, code_size), device=gt_boxes.device)

        assert gt_boxes.shape[0] == pseudo_boxes.shape[0], f'batch size should be the same, gt:{gt_boxes.shape} pseudo:{pseudo_boxes.shape}'
        for b in range(gt_boxes.shape[0]):
            cur_gt_boxes = valid_boxes(gt_boxes[b])
            cur_pseudo_boxes = valid_boxes(pseudo_boxes[b])
            num_gt = cur_gt_boxes.shape[0]
            num_pseudo = cur_pseudo_boxes.shape[0]

            num = num_gt + num_pseudo

            if num > MN:
                MN = num

            ret_boxes[b, :num_gt] = cur_gt_boxes
            pseudo_dim = cur_pseudo_boxes.shape[-1]
            ret_boxes[b, num_gt:num, :(pseudo_dim - 1)] = cur_pseudo_boxes[..., :-1]
            ret_boxes[b, num_gt:num, -1] = cur_pseudo_boxes[..., -1]

        ret_boxes = ret_boxes[:, :MN].contiguous() # remove redundant padding

        return ret_boxes

    def __call__(self, batch_dict):
        """
        Args:
            batch_dict:
                gt_boxes: gt_boxes of known categories
        Returns:
            batch_dict:
                gt_boxes: gt_boxes of knowns, with pseudo labels of unknowns
        """
        if not self.training: 
            return batch_dict

        gt_boxes = batch_dict['gt_boxes']
        pseudo_boxes = batch_dict['pseudo_boxes']

        print('gt', gt_boxes.shape, 'pseudo', pseudo_boxes.shape)

        # relabel gt to full class labels and then combine with pseudo labels
        gt_boxes = self.relabel_gt_boxes(gt_boxes)
        # pseudo_boxes = self.load_pseudos(batch_dict)

        # pseudo sampling to get more boxes for unknowns
        # pseudo_boxes = self.sampler(batch_dict, pseudo_boxes, gt_boxes)

        gt_boxes = self.combine_gt_with_pseudos(gt_boxes, pseudo_boxes)
        # print('combined boxes', gt_boxes)

        batch_dict['gt_boxes'] = gt_boxes

        return batch_dict
