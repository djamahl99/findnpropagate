from pathlib import Path
from typing import Any
import torch
from torch import Tensor
import numpy as np

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import common_utils
from pcdet.models.dense_heads.target_assigner.hungarian_assigner import HungarianAssigner3D

from typing import Dict, List
import os

def valid_boxes(cur_gt_bboxes_3d: Tensor) -> list:
    valid_idx = []
    # filter empty boxes
    for i in range(len(cur_gt_bboxes_3d)):
        if cur_gt_bboxes_3d[i][3] > 0 and cur_gt_bboxes_3d[i][4] > 0: # size > 0
            valid_idx.append(i)

    return cur_gt_bboxes_3d[valid_idx]

def count_classes(curr_boxes: Tensor, curr_stats: Dict, all_class_names: list) -> Dict:
    for box in curr_boxes:
        lbl = int(box[..., -1].long().item() - 1)
        cls_name = all_class_names[lbl]

        curr_stats[f'num_per_class_{cls_name}'] += 1

    return curr_stats

# def batch_apply(data, preds_dicts, required_keys, apply_fn):
#     assert all([k in data.keys() for k in required_keys])

#     for b in range(len(preds_dicts)):
#         preds_boxes = preds_dicts[b]['pred_boxes']

#         batch_data = {k: data[k][b] for k in required_keys}

#         preds_boxes = apply_fn(batch_data, preds_boxes)
        
#         preds_dicts[b]['pred_boxes'] = preds_boxes

#     return preds_dicts 

def single_batch_apply(batch_dict: Dict, preds_dict: Dict, required_keys: List, apply_fn, b: int):
    assert all([k in batch_dict.keys() for k in required_keys])
    
    batch_data = {k: batch_dict[k][b] for k in required_keys}

    pred_boxes = apply_fn(batch_data, preds_dict['pred_boxes'])
    preds_dict['pred_boxes'] = pred_boxes

    return preds_dict

class AugReverse:
    @staticmethod
    def random_world_flip(data, preds_dict, b):
        def apply_fn(batch_data, pred_boxes):
            cur_flip_x = batch_data['flip_x']
            cur_flip_y = batch_data['flip_y']

            if cur_flip_x:
                pred_boxes[..., 1] = -pred_boxes[..., 1]
                pred_boxes[..., 6] = -pred_boxes[..., 6]

            if cur_flip_y:
                pred_boxes[..., 0] = -pred_boxes[..., 0]
                pred_boxes[..., 6] = -(pred_boxes[..., 6] + np.pi)
            
            return pred_boxes

        return single_batch_apply(data, preds_dict, ['flip_x', 'flip_y'], apply_fn, b)

    @staticmethod
    def random_world_rotation(data, preds_dict, b):
        def apply_fn(batch_data, pred_boxes):
            cur_noise_rotation = batch_data['noise_rot'].to(pred_boxes.device)

            pred_boxes[:, 0:3] = common_utils.rotate_points_along_z(pred_boxes[None, :, 0:3], -cur_noise_rotation[None])[0]
            pred_boxes[:, 6] -= cur_noise_rotation
            
            return pred_boxes

        return single_batch_apply(data, preds_dict, ['noise_rot'], apply_fn, b)

    @staticmethod
    def random_world_scaling(data, preds_dict, b):
        def apply_fn(batch_data, pred_boxes):
            cur_noise_scale = batch_data['noise_scale'].to(pred_boxes.device)

            pred_boxes[:, 0:3] /= cur_noise_scale
            pred_boxes[:, 6] /= cur_noise_scale
            
            return pred_boxes

        return single_batch_apply(data, preds_dict, ['noise_scale'], apply_fn, b)

    @staticmethod
    def random_world_translation(data, preds_dict, b):
        def apply_fn(batch_data, pred_boxes):
            cur_translation = batch_data['noise_translate'].to(pred_boxes.device)

            pred_boxes[:, 0:3] -= cur_translation
            
            return pred_boxes

        return single_batch_apply(data, preds_dict, ['noise_translate'], apply_fn, b)

class PseudoProcessor(object):
    self_training: bool
    self_training_folder: str
    sample_iou_thresh: float = 0.01
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    cons_iou_thresh: float = 0.3
    """
    Loads pseudo labels for unknown classes in current batch and combines them with gt knowns.
    """
    def __init__(self, known_class_names: list, self_training_folder: str = None, all_class_names: list=None):
        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        if all_class_names is not None:
            self.all_class_names = all_class_names
        self.known_class_names = known_class_names
        self.num_classes = len(self.all_class_names)

        # whether we are doing self-training and will save boxes live
        self.self_training = self_training_folder is not None

        self.is_known = {i: (cls in self.known_class_names) for i, cls in enumerate(self.all_class_names)}

        # in open vocab, training has less classes
        # self.training = len(self.known_class_names) != self.num_classes and 
        self.training = set(self.known_class_names) != set(self.all_class_names)
        
        # gt_labels are 1 indexed
        self.gt_known_to_full_labels = {(i + 1): (j + 1) for i, known_name in enumerate(self.known_class_names) for j, all_name in enumerate(self.all_class_names) if known_name == all_name}
        self.full_labels_to_gt_known = {v: k for k, v in self.gt_known_to_full_labels.items()}
        self.unknown_labels = [(i + 1) for i, cls_name in enumerate(self.all_class_names) if cls_name not in self.known_class_names]
        self.all_labels = [(i + 1) for i, _ in enumerate(self.all_class_names)]


        if self.training: # print unknown information (in testing we have the gt for all boxes)
            for i, cls in enumerate(self.all_class_names):
                mp_str = ""
                if (i + 1) in self.full_labels_to_gt_known:
                    mp_str = f"no. {self.full_labels_to_gt_known[i+1]} => {i+1} full labels"
                    mp_str += f", {self.all_class_names[i]} => {self.known_class_names[self.full_labels_to_gt_known[i + 1] - 1]}"
                print(f"{cls}: {'Known' if self.is_known[i] else 'Unknown'}", mp_str)

            print('unknown labels', self.unknown_labels)

        self.pseudos_missing = set()

        if self.self_training:
            self.self_training_folder = self_training_folder
            parent_folder = Path(self.self_training_folder).parent
            assert os.path.exists(parent_folder), f'self training folder parent should exist! {parent_folder}'

            if not os.path.exists(self.self_training_folder):
                os.makedirs(self.self_training_folder)

        # logging
        self.forward_pseudo_stats = {}

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

    def combine_gt_with_pseudos(self, gt_boxes: Tensor, pseudo_boxes: Tensor) -> Tensor:
        """Combine gt boxes for known classes with pseudo boxes for unknowns

        Args:
            gt_boxes (Tensor): gt boxes (B, N, 8) from dataloader, relabeled by relabel_gt_boxes
            pseudo_boxes (Tensor): pseudo boxes (B, M, 8)

        Returns:
            Tensor: combined boxes (B, N+M, 8)
        """
        # for logging
        curr_stats = {'num_gt': 0, 'num_pseudo': 0}
        for cls_name in self.all_class_names:
            curr_stats[f'num_per_class_{cls_name}'] = 0

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

            # STATS LOGGING ############################
            curr_stats['num_gt'] += num_gt
            curr_stats['num_pseudo'] += num_pseudo

            curr_stats = count_classes(cur_gt_boxes, curr_stats, self.all_class_names)
            curr_stats = count_classes(cur_pseudo_boxes, curr_stats, self.all_class_names)
            # STATS LOGGING ############################

            num = num_gt + num_pseudo

            if num > MN:
                MN = num

            ret_boxes[b, :num_gt] = cur_gt_boxes
            pseudo_dim = cur_pseudo_boxes.shape[-1]
            ret_boxes[b, num_gt:num, :(pseudo_dim - 1)] = cur_pseudo_boxes[..., :-1]
            ret_boxes[b, num_gt:num, -1] = cur_pseudo_boxes[..., -1]

        ret_boxes = ret_boxes[:, :MN].contiguous() # remove redundant padding

        # STATS
        for k in curr_stats.keys():
            curr_stats[k] = curr_stats[k] / max(gt_boxes.shape[0], 1)

            self.forward_pseudo_stats[k] = curr_stats[k]

        # self.forward_pseudo_stats = curr_stats

        return ret_boxes

    def undo_augmentations(self, batch_dict: Dict, preds_dict: Dict, b: int) -> Dict:
        """Undos augmentations before saving predictions.

        Args:
            batch_dict (Dict): _description_
            preds_dict (Dict): _description_
            b (int): batch idx

        Returns:
            Dict: transformed preds_dict
        """
        AUG_DATA_MAP = {
            'random_world_flip': ['flip_x', 'flip_y'],
            'random_world_rotation': ['noise_rot'],
            'random_world_scaling': ['noise_scale'],
            'random_world_translation': ['noise_translate']
        }
        AUG_ORDER = ['random_world_flip', 'random_world_rotation', 'random_world_scaling', 'random_world_translation']

        for aug in reversed(AUG_ORDER):
            data = {}
            in_batch = False
            # data = {k: batch_dict[k] for k in AUG_DATA_MAP[aug] if k in batch_dict}

            for k in AUG_DATA_MAP[aug]:
                if k in batch_dict:
                    data[k] = batch_dict[k]
                    in_batch = True

            if in_batch:
                preds_dict = getattr(AugReverse, aug)(data, preds_dict, b)

        return preds_dict
    
    def save_predictions(self, batch_dict: Dict, preds_dicts: Dict, epoch: int=0) -> None:
        """Saves prediction for self-training.

        Args:
            batch_dict (Dict): _description_
            preds_dicts (Dict): _description_
        """
        pseudo_boxes = batch_dict.get('pseudo_boxes', None)
        sample_mask = batch_dict.get('pseudo_samples_mask', None)

        # if 'pseudo_samples_mask' in batch_dict:
        #     sample_mask = batch_dict['pseudo_samples_mask']
        # else:
        #     sample_mask = pseudo_boxes.new_zeros(pseudo_boxes.shape[0], dtype=torch.bool)


        pred_keys = set(['pred_boxes', 'pred_scores', 'pred_labels'])

        assigner = HungarianAssigner3D({}, {}, {})

        batch_cons_sum = {l: 0.0 for l in self.all_labels}

        for b, (frame_id, preds_dict) in enumerate(zip(batch_dict['frame_id'], preds_dicts)):

            for k in preds_dict.keys():
                if k not in pred_keys:
                    continue
                preds_dict[k] = preds_dict[k].detach().clone().cpu()

            if pseudo_boxes is not None:
                sample_mask = sample_mask.to(dtype=torch.bool)

                cur_pseudo_boxes = pseudo_boxes[b]
                cur_sample_mask = sample_mask[b]
                sampled_boxes = cur_pseudo_boxes[cur_sample_mask].cpu()

                if sampled_boxes.shape[0] > 0 and preds_dict['pred_boxes'].shape[0] > 0:
                    # remove those that overlap with the copy and pasted objects
                    ious = iou3d_nms_utils.boxes_bev_iou_cpu(preds_dict['pred_boxes'][:, :7], sampled_boxes[:, :7])
                    ious = ious.max(dim=1).values

                    valid_mask = (ious < self.sample_iou_thresh) # was 0.4
                    for k in preds_dict.keys():
                        if k not in pred_keys:
                            continue
                        preds_dict[k] = preds_dict[k][valid_mask]

            # undo augmentations before saving
            preds_dict = self.undo_augmentations(batch_dict, preds_dict, b)

            pseudo_path = Path(self.self_training_folder) / f"{frame_id.replace('.', '_')}.pth"

            # consistency?
            if pseudo_path.exists():
                curr_boxes = preds_dict['pred_boxes']
                curr_labels = preds_dict['pred_labels']

                cons_sum = {l: 0 for l in self.all_labels}
                try:
                    # load previous round preds
                    old_dict = torch.load(pseudo_path, map_location='cpu')

                    old_boxes = old_dict['pred_boxes']
                    old_labels = old_dict['pred_labels']

                    overlaps = iou3d_nms_utils.boxes_bev_iou_cpu(curr_boxes[:, :7], old_boxes[:, :7])

                    overlaps = overlaps.max(dim=1).values

                    # consistency
                    # cons_sum = (overlaps >= self.cons_iou_thresh).sum().item()

                    cons_bool = (overlaps >= self.cons_iou_thresh)
                    for lbl, cons in zip(curr_labels, cons_bool):
                        if cons:
                            cons_sum[int(lbl)] += 1 
                except Exception as e:
                    print('Exception when trying to calculate consistency with hungarian assigner =>', e)

                for l in self.all_labels:
                    batch_cons_sum[l] += cons_sum[l]

            preds_dict['epoch'] = epoch

            # save this iteration
            torch.save(preds_dict, pseudo_path)          

        # STATS
        # mean_cons = batch_cons_sum / batch_dict['batch_size']  
        mean_cons = {l: batch_cons_sum[l] / batch_dict['batch_size'] for l in self.all_labels}

        # self.forward_pseudo_stats['mean_consistent'] = mean_cons

        for l in self.all_labels:
            cls_name = self.all_class_names[int(l - 1)]
            self.forward_pseudo_stats[f'mean_consistent_{cls_name}'] = mean_cons[l]

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

        # relabel gt to full class labels and then combine with pseudo labels
        gt_boxes = self.relabel_gt_boxes(gt_boxes)
        gt_boxes = self.combine_gt_with_pseudos(gt_boxes, pseudo_boxes)

        if 'pseudo_samples_mask' in batch_dict:
            self.forward_pseudo_stats['mean_samples'] = batch_dict['pseudo_samples_mask'].sum(dim=1).mean()
        else:
            self.forward_pseudo_stats['mean_samples'] = 0

        batch_dict['gt_boxes'] = gt_boxes

        return batch_dict
