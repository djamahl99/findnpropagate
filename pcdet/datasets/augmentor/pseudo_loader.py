from pathlib import Path
from typing import Any, Tuple
import torch
from torch import Tensor
import numpy as np

from typing import Dict, List
import os

from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils.common_utils import rotate_points_along_z

def remove_empty(pseudo_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove empty boxes (those that have 0m length / width)

    Args:
        pseudo_boxes (np.ndarray): boxes to filter

    Returns:
        np.ndarray: filtered boxes
        np.ndarray: filter mask
    """
    non_empty = np.bitwise_and(pseudo_boxes[:, 3] > 0, pseudo_boxes[:, 4] > 0)
    pseudo_boxes = pseudo_boxes[non_empty]

    return pseudo_boxes, non_empty

def bev_nms_cpu(boxes: torch.tensor, scores: torch.tensor, thresh: float = 0.5):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    N = boxes.shape[0]
    indices = torch.arange(0, N)
    keep = torch.ones_like(indices, dtype=torch.bool)
    order = torch.argsort(- scores)

    ious = iou3d_nms_utils.boxes_bev_iou_cpu(boxes, boxes)

    for i in range(N):
        if keep[i]:
            curr_idx = order[i]

            for j in range(i + 1, N):
                other_idx = order[j]

                bev_iou = ious[curr_idx, other_idx]

                if bev_iou > thresh:
                    keep[j] = 0

    return order[keep]

class ObjectSample(object):
    num_points: int
    points: np.ndarray
    box: np.ndarray
    label: int
    l: float
    w: float
    h: float
    x: float
    y: float
    z: float
    ry: float
    conf: float

    def __init__(self, relative_points: np.ndarray, box: np.ndarray, conf: float) -> None:
        self.conf = conf
        self.num_points = relative_points.shape[0]
        self.label = box[..., -1].item() 

        box = box.copy()
        box = box.reshape(1, 8)

        self.l, self.w, self.h = [x.item() for x in box[0, [3, 4, 5]]]
        self.x, self.y, self.z = [x.item() for x in box[0, 0:3]]
        self.ry = box[:, 6].item()

        # now centre box
        box[:, 0:3] = 0
        # remove rotation
        box[:, 6] = 0

        # points, box = self.transform_to_axis_aligned(points, box)
        self.points, self.box = relative_points, box
    
    def transform_to_axis_aligned(self, points: np.ndarray, box3d: np.ndarray) -> np.ndarray:
        """Translates and then rotates points to the box coordinate frame.

        Args:
            points (np.ndarray): (N, 3) points
            box3d (np.ndarray): (1, 7 + C) box

        Returns:
            np.ndarray: (N, 3) translated + rotated points
        """
        points = points.clone()
        points = points[:, :3] - box3d[:, 0:3] # centre the points
        # points = points.permute(1, 0, 2)

        points = rotate_points_along_z(points[None], - box3d[:, 6]).view(-1, 3) # rotate by negative angle

        # now centre box
        box3d[:, 0:3] = 0
        # remove rotation
        box3d[:, 6] = 0

        return points, box3d
    
    def dropout_points(self, dropout: float=0.5, min_points: int=5) -> np.ndarray:
        """
            Returns points from this object with some dropped out, according to dropout.

        Returns:
            np.ndarray: remaining points after dropout
        """
        if self.points.shape[0] <= min_points*2:
            return self.points.copy()
        
        points = self.points.copy()

        if np.random.rand() < dropout:
            # drop out
            num_pts = len(points)
            # num_to_keep = np.random.randint(min_points, len(points))
            num_to_keep = np.random.randint(num_pts//2, num_pts)
            idx = np.random.randint(0, len(points), size=num_to_keep)
            points = points[idx]

        return points

    def get_sample_points(self, sample_box: np.ndarray, dropout: float=0.5) -> np.ndarray:
        """Transform our box points to the coordinates relative to sample_box/

        Args:
            sample_box (np.ndarray): the sampled box to transform the points to
            dropout (float, optional): the probability of dropping out some of the points. Defaults to 0.5.

        Returns:
            np.ndarray: sampled points for the sample box
        """
        points = self.dropout_points(dropout)

        points = rotate_points_along_z(points[None], sample_box[:, 6]).reshape(-1, 5) # rotate
        box_centres_repeated = np.repeat(sample_box[:, 0:3], repeats=(points.shape[0]), axis=0) # translate
        points[:, :3] += box_centres_repeated

        return points

    def sample(self, gt_boxes_tensor: Tensor, pseudo_boxes: np.ndarray, 
               max_iou: float=0.1, dropout: float=0.5, min_dist: float=4.5, 
               rot_noise: float=np.pi/4.0, trans_noise: float=2.0) -> tuple:
        """ Sample from this object
            - adds random x, y and then infers z from gt? and makes sure that we are not overlapping with other boxes
        """
        # obj_bottoms = (gt_boxes[..., 2] - gt_boxes[..., 5] / 2)
        # bot_min, bot_max = obj_bottoms.min(), obj_bottoms.max()
        # bev_min, bev_range = -50.0, 100.0

        valid = False

        max_tries = 10
        tries = 0
        
        while not valid and tries < max_tries:
            tries += 1

            X, Y, Z = np.random.randn((3)) # add x, y, z shift (as otherwise invalidates timestamp etc)
            x = self.x + trans_noise * X
            y = self.y + trans_noise * Y
            z = self.z + trans_noise * Z


            dist = np.linalg.norm([x, y, z])
            if dist < min_dist:
                continue

            alpha = self.ry + rot_noise * np.random.rand() # add some rotation change (otherwise seen face is not facing ego vehicle)

            sampled_box = torch.tensor([x, y, z, self.l, self.w, self.h, alpha, self.label], dtype=torch.float32).reshape(1, 8)

            if gt_boxes_tensor.shape[0] == 0 and pseudo_boxes.shape[0] == 0:
                valid = True
                continue

            # check not overlapping with gt
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_box[:, :7], gt_boxes_tensor)

            if ious.shape[0] > 0 and ious.max() >= max_iou:
                continue

            if pseudo_boxes.shape[0] == 0:
                valid = True
                continue

            # check not overlapping with pseudo
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_box[:, :7], pseudo_boxes[:, :7])

            if ious.size == 0:
                valid = True
                continue

            if ious.max() < max_iou:
                valid = True

        if not valid: # ran out of tries
            return None, None

        return sampled_box.numpy(), self.get_sample_points(sampled_box.numpy(), dropout=dropout)

    def __repr__(self) -> str:
        return f"ObjectSample[Class: {self.label}] ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})) {self.ry:.2f}) [{self.l:.2f}, {self.w:.2f}, {self.h:.2f}] {self.num_points} points"

class PseudoSampler(object):
    unknown_queue: Dict[int, List[ObjectSample]]
    unknown_class_labels: list
    known_class_labels: list
    max_queue_size_per_class: int
    prop_per_unk: Dict[int, float] # includes all class
    mom: float = 0.9
    num_classes: int # unkn + known
    dropout: float = 0.5
    min_pts: int = 5 # minimum pts to consider box
    min_dist: float = 3.0 # minimum distance to be a valid sample
    pseudo_nms_thresh: float = 1e-7
    known_to_unknown_ratio: float
    ego_vehicle: Tensor = None
    queue_metric = 'num_pts'
    rot_noise: float = np.pi/4.0
    trans_noise: float = 2.0
    validate_pseudos: bool = True
    timestamp= None

    def __init__(self, class_labels: list, known_class_labels: list, unknown_class_labels: list, 
                 max_queue_size_per_class: int = 100, num_classes: int = 10, 
                 dropout: float = 0.5, mom: float=0.9) -> None:
        self.known_class_labels = known_class_labels
        self.unknown_class_labels = unknown_class_labels
        self.max_queue_size_per_class = max_queue_size_per_class
        self.class_labels = class_labels
        
        self.dropout = dropout
        self.mom = mom

        # initialize queue
        self.unknown_queue = {l: [] for l in self.unknown_class_labels}
        self.num_classes = num_classes

        # uniform init
        self.prop_per_unk = {idx: 1.0/float(len(self.unknown_class_labels)) for idx in self.unknown_class_labels}

        self.known_to_unknown_ratio = (len(self.unknown_class_labels)) / (num_classes - len(self.unknown_class_labels) + 1e-6)

    def calc_seen_per_class(self, pseudo_boxes: np.ndarray, gt_boxes: np.ndarray) -> None:
        pseudo_labels = pseudo_boxes[..., -1].reshape(-1).astype(np.int32)

        # boxes_curr = (pseudo_labels > 0).sum()
        pseudo_num = float(max(pseudo_labels.size, 1e-7))

        for idx in self.unknown_class_labels:
            num_curr = (pseudo_labels == idx).sum()

            # EMA proportion per unknown
            self.prop_per_unk[idx] = self.prop_per_unk[idx] * self.mom + (num_curr / pseudo_num) * (1.0 - self.mom)

    def points_in_boxes(self, points: np.ndarray, boxes3d: np.ndarray) -> tuple:
        N = boxes3d.shape[0]

        points = points.copy()
        boxes3d = boxes3d.copy()
        template = np.array([
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        ]) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(repeats=8, axis=1) * template[None, :, :]
        # instead of rotating the box, rotate the pointcloud (so can simply threshold the points)
        points = points[:, None, :].repeat(repeats=N, axis=1)
        points[..., :3] = points[..., :3] - boxes3d[None, :, 0:3] # centre the points
        points = points.transpose((1, 0, 2))

        points = rotate_points_along_z(points, - boxes3d[:, 6]) # rotate by negative angle

        x1, x2 = np.min(corners3d[..., 0], axis=1, keepdims=True), np.max(corners3d[..., 0], axis=1, keepdims=True)
        y1, y2 = np.min(corners3d[..., 1], axis=1, keepdims=True), np.max(corners3d[..., 1], axis=1, keepdims=True)
        z1, z2 = np.min(corners3d[..., 2], axis=1, keepdims=True), np.max(corners3d[..., 2], axis=1, keepdims=True)

        in_box = (
            (points[..., 0] >= x1) &
            (points[..., 0] <= x2) &
            
            (points[..., 1] >= y1) &
            (points[..., 1] <= y2) &
            
            (points[..., 2] >= z1) &
            (points[..., 2] <= z2)
            )

        assert points.shape[-1] == 5


        # timestamps = points[..., 4]
        # print('timestamps', timestamps.min(), timestamps.max())

        # # check timestamp
        # if self.timestamp is not None:
        #     in_box = in_box & (timestamps == 0)
        #     print()
        # else:
        #     print('timestamp none')


        return in_box, points

    def __call__(self, batch_dict, pseudo_boxes: np.ndarray, pseudo_scores: np.ndarray, gt_boxes: np.ndarray, sample_buffer_num: int=5, fix_cp: int=None) -> Dict:
        self.calc_seen_per_class(pseudo_boxes, gt_boxes)

        # number of samples we have collected
        samples_per_label = {l: len(x) for l, x in self.unknown_queue.items()}

        num_gt_cls_scaled = int(gt_boxes.shape[0]*self.known_to_unknown_ratio)
        num_scaled = max(num_gt_cls_scaled, pseudo_boxes.shape[0])

        num_proposals = num_scaled + sample_buffer_num
        
        if fix_cp is not None:
            num_proposals = num_scaled + fix_cp

        # add base points
        cur_points = batch_dict['points'].copy()
        batch_points = [cur_points]

        if pseudo_boxes.size == 0:
            return pseudo_boxes, np.zeros((0), dtype=bool)

        # find pseudos that don't overlap with GT (already done in load_frustum_pseudos)
        # pseudo_boxes_tensor = torch.tensor(pseudo_boxes.copy(), dtype=torch.float32)
        gt_boxes_tensor = torch.tensor(gt_boxes.copy(), dtype=torch.float32)

        # add ego to gt to remove any samples that overlap
        gt_boxes_tensor = torch.cat((gt_boxes_tensor[:, :7], self.ego_vehicle.to(gt_boxes_tensor.device)), dim=0)

        points_in_boxes, box_relative_pts = self.points_in_boxes(cur_points[:, :], pseudo_boxes[:, :7])

        # keep track of number of samples per class, and those with enough points to be valid
        num_pseudos = pseudo_boxes.shape[0]
        curr_num_per_class = {l: 0 for l in self.unknown_class_labels}
        valid_idx = []

        num_points_per_box = points_in_boxes.sum(axis=1)

        if self.queue_metric == 'num_pts':
            idx_sorted = np.argsort(-num_points_per_box, axis=0)
        else:
            idx_sorted = np.argsort(-pseudo_scores, axis=0)
        # idx_sorted = torch.argsort(num_points_per_box, descending=True)

        # don't want more than max_num_per_unknown for unknown classes (e.g. lots of unknown due to sampling +  etc)
        max_num_per_unknown = gt_boxes.shape[0] / max(len(self.known_class_labels), 1)

        # collect object samples
        # for idx, box in enumerate(pseudo_boxes):
        for idx in idx_sorted:
            box = pseudo_boxes[idx]
            lbl = int(box[..., -1])
            curr_box = box.copy()
            pt_mask = points_in_boxes[idx]
            # pseudo_points = cur_points[pt_mask]
            pseudo_rel_points = box_relative_pts[idx, pt_mask]

            if not self.validate_pseudos:
                valid_idx.append(idx)

            # check if has enough points to add to queue
            if pseudo_rel_points.shape[0] < self.min_pts:
                continue

            # check if is not too close to ego
            if np.linalg.norm(curr_box[:3]) < self.min_dist:
                # probably too close
                continue

            curr_num_per_class[lbl] += 1

            if self.validate_pseudos:
                valid_idx.append(idx)

            curr_conf = pseudo_scores[idx]

            if samples_per_label[lbl] >= self.max_queue_size_per_class:
                # find worst sample (e.g. worst conf)
                if self.queue_metric == 'num_pts':
                    queue_pts = np.array([x.num_points for x in self.unknown_queue[lbl]])

                    rpl_idx = np.argmin(queue_pts)

                    self.unknown_queue[lbl][rpl_idx] = ObjectSample(pseudo_rel_points, curr_box, conf=curr_conf)
                else:
                    queue_confs = np.array([x.conf for x in self.unknown_queue[lbl]])
                    rpl_idx = np.argmin(queue_confs)
                    rpl_conf = queue_confs[rpl_idx]

                    if curr_conf > rpl_conf:
                        self.unknown_queue[lbl][rpl_idx] = ObjectSample(pseudo_rel_points, curr_box, conf=curr_conf)
            else:
                self.unknown_queue[lbl].append(ObjectSample(pseudo_rel_points, curr_box, conf=curr_conf))

        # number of valid pseudo boxes
        num_pseudos = len(valid_idx)

        # out array
        pseudos_out = np.zeros((num_proposals, 8))
        pseudos_out[:num_pseudos] = pseudo_boxes[valid_idx]

        # whether each box is a sample
        sample_mask = np.zeros((num_proposals), dtype=bool)

        # number of objects to sample
        num_samples = max(num_proposals - num_pseudos, 0)

        # fixed number of copy-paste objects
        if fix_cp is not None:
            num_samples = fix_cp

        # no sampling is to be done
        if num_samples <= 0 or max(samples_per_label) == 0:
            pseudos_out = pseudos_out[:num_pseudos]
            sample_mask = sample_mask[:num_pseudos]

            return pseudos_out, sample_mask

        # 1 / num of unknowns
        unif_prob = (1.0/float(len(self.unknown_class_labels)))

        # num_samples / (num unknowns) - (num seen of this class)
        # sample_num_per_class = {l: max(unif_prob*num_samples - curr_num_per_class[l], 0.0) for l in self.unknown_class_labels}

        # set idx to where pseudo boxes end
        sample_idx = num_pseudos

        curr_sampled = {l: 0 for l in self.unknown_class_labels}

        # add some samples
        # for lbl in labels_to_sample:
        for i in range(num_samples):
            lbl = np.random.choice(self.unknown_class_labels)

            if samples_per_label[lbl] == 0: # have not collected any for this class yet
                continue

            # skip this label if already have plenty of samples for it
            if (curr_num_per_class[lbl] + curr_sampled[lbl]) >= max_num_per_unknown:
                continue

            # choose which box in the queue for this label to sample from
            idx = np.random.choice(len(self.unknown_queue[lbl]))
            sample_box, sample_points = self.unknown_queue[lbl][idx].sample(gt_boxes_tensor, pseudos_out[:sample_idx], min_dist=self.min_dist, 
                                                                            rot_noise=self.rot_noise, trans_noise=self.trans_noise)

            # sampling was unsuccessful (collision etc)
            if sample_box is None:
                continue

            # add to pseudos list
            pseudos_out[sample_idx] = sample_box
            sample_mask[sample_idx] = True

            # increase sampled number for this label
            curr_sampled[lbl] += 1

            # increase index of out array
            sample_idx += 1

            # add the points to our points list
            batch_points.append(sample_points)

        # add all the sampled points
        batch_points = np.concatenate(batch_points, axis=0)
        batch_dict['points'] = batch_points

        return pseudos_out, sample_mask

class PseudoLoader(object):
    dropout: float
    min_score: float
    unknown_score_ema: Dict[int, float]
    mom: float = 0.9
    pseudo_nms_thresh: float
    ego_vehicle: Tensor
    max_num_gt_class: float = 20.0
    max_selftrain_per_class: int = None
    copy_st_only: bool = False
    """
    Loads pseudo labels for unknown classes in current batch and combines them with gt knowns.
    """
    def __init__(self, known_class_names: list, pseudo_path='pseudo_labels/frustum_proposals/', self_train_path: str = None, 
                 dropout: float=0.5, min_score: float=0.1, pseudo_nms_thresh: float=1e-7, max_selftrain_per_class: int = None, 
                 fix_cp: int = None, mom: float=0.9, copy_st_only: bool = False, sampler_val: bool=True):
        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = known_class_names
        self.num_classes = len(self.all_class_names)
        self.max_selftrain_per_class = max_selftrain_per_class
        self.fix_cp = fix_cp
        self.mom = mom
        self.copy_st_only = copy_st_only

        # in open vocab, training has less classes
        self.training = len(self.known_class_names) != self.num_classes

        self.class_labels = [i for i in range(1, self.num_classes + 1)]
        self.unknown_class_labels = [(i + 1) for i, cls_name in enumerate(self.all_class_names) if cls_name not in self.known_class_names]
        self.known_class_labels = [i for i in self.class_labels if i not in self.unknown_class_labels]
        self.is_known = {i: (i in self.known_class_labels) for i in self.class_labels}

        assert len(set(self.unknown_class_labels).union(self.known_class_labels)) == self.num_classes, \
            f'error, known + unkwnown == num_classes knowns={self.known_class_labels} unknowns={self.unknown_class_labels}'
        assert set(self.unknown_class_labels).union(self.known_class_labels) == set(self.class_labels), \
            f'error, known_labels + unkwnown_labels != class_labels knowns={self.known_class_labels} unknowns={self.unknown_class_labels}, all={self.class_labels}'

        self.ego_vehicle = torch.tensor([[0, -1.0, (-5.0 + 3.0)/2.0, 5.0, 3.0, 8.0, np.pi/2.0]], dtype=torch.float32, device='cpu')

        # gt_labels are 1 indexed
        self.gt_known_to_full_labels = {(i + 1): (j + 1) for i, known_name in enumerate(self.known_class_names) 
                                        for j, all_name in enumerate(self.all_class_names) if known_name == all_name}
        self.full_labels_to_gt_known = {v: k for k, v in self.gt_known_to_full_labels.items()}
        self.unknown_class_labels = [(i + 1) for i, cls_name in enumerate(self.all_class_names) if cls_name not in self.known_class_names]
        self.unknown_score_ema = {l: min_score for l in self.unknown_class_labels}
        
        # if self.training: # print unknown information (in testing we have the gt for all boxes)
        #     for i, cls in enumerate(self.all_class_names):
        #         mp_str = ""
        #         if (i + 1) in self.full_labels_to_gt_known:
        #             mp_str = f"no. {self.full_labels_to_gt_known[i+1]} => {i+1} full labels"
        #             mp_str += f", {self.all_class_names[i]} => {self.known_class_names[self.full_labels_to_gt_known[i + 1] - 1]}"
        #         print(f"{cls}: {'Known' if self.is_known[i] else 'Unknown'}", mp_str)

        #     print('unknown labels', self.unknown_class_labels)

        self.pseudos_missing = set()

        self.dropout = dropout
        self.min_score = min_score
        self.pseudo_nms_thresh = pseudo_nms_thresh
        self.pseudo_folder = pseudo_path
        self.self_training_folder = self_train_path

        self.copy_boxes, self.copy_scores, self.pseudo_types = None, None, None

        self.sampler = PseudoSampler(class_labels=self.class_labels, known_class_labels=self.known_class_labels, 
                                     unknown_class_labels=self.unknown_class_labels, max_queue_size_per_class=100, 
                                     dropout=dropout, mom=mom)
        self.sampler.pseudo_nms_thresh = pseudo_nms_thresh
        self.sampler.ego_vehicle = self.ego_vehicle
        self.sampler.validate_pseudos = sampler_val

    def load_pseudos(self, batch_dict, unknowns_only=True, folder: str = None, 
                     record_missing: bool = True, filter_by_score: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Load pseudo labels

        Args:
            batch_dict (dict): batch data
            unknowns_only (bool): only return pseudo boxes for unknowns
            
        Returns:
            np.ndarray: pseudo boxes (M, 8)
            np.ndarray: pseudo scores (M)
        """
        frame_id = batch_dict['frame_id']

        if folder is None:
            folder = self.pseudo_folder

        pseudo_path = Path(folder) / f"{frame_id.replace('.', '_')}.pth"

        if not os.path.exists(pseudo_path):
            if record_missing:
                self.pseudos_missing.add(str(pseudo_path))
                print('psuedos dont exist!', self.pseudos_missing)

            return np.zeros((0, 8)), np.zeros((0))

        try:
            preds_dicts = torch.load(pseudo_path, map_location='cpu')
        except Exception as e:
            print('Exception occured when loading self-training preds =>', e)
            return np.zeros((0, 8)), np.zeros((0))

        assert len(preds_dicts) == 1 or isinstance(preds_dicts, dict), f"preds dict should have len==1, got {len(preds_dicts)} {type(preds_dicts)}"

        if len(preds_dicts) == 1:
            preds_dict = preds_dicts[0]
        elif isinstance(preds_dicts, dict):
            preds_dict = preds_dicts # no batch

        pred_boxes = preds_dict['pred_boxes'].numpy()
        pred_scores = preds_dict['pred_scores'].numpy()
        pred_labels = preds_dict['pred_labels'].numpy()

        unknown_threshs = {l: 0.0 for l in self.unknown_class_labels}
        num_per_unk = {l: 0 for l in self.unknown_class_labels}

        # use this for limiting number of pseudos
        # gt_names = batch_dict['gt_names']
        # max_num_gt_class = 0
        # for cls_name in self.all_class_names:
        #     num_curr = int((gt_names == cls_name).sum())

        #     if num_curr > max_num_gt_class:
        #         max_num_gt_class = num_curr

        # # ema update
        # self.max_num_gt_class = self.mom * self.max_num_gt_class + (1.0 - self.mom) * max_num_gt_class

        if self.max_selftrain_per_class is not None: 
            for lbl in self.unknown_class_labels:
                label_scores = pred_scores[pred_labels == lbl]

                num_per_unk[lbl] += label_scores.size

                if label_scores.size == 0:
                    continue
                elif label_scores.size < self.max_selftrain_per_class:
                    unknown_threshs[lbl] = np.min(label_scores)
                else:
                    # use max_selftrain_per_class'th highest score as a second threshold
                    order = np.argsort(-label_scores, axis=0)
                    topkidx = order[int(min(self.max_selftrain_per_class, order.shape[0]) - 1)]
                    unknown_threshs[lbl] = float(label_scores[topkidx])

        num_after = {l: 0 for l in self.unknown_class_labels}
        if unknowns_only: # don't need pseudo labels for knowns
            pred_mask = np.zeros_like(pred_labels, dtype=bool)
            for i, lbl in enumerate(pred_labels):
                # pred mask
                pred_mask[i] = lbl in self.unknown_class_labels

                if pred_mask[i]:
                    # filter by above EMA
                    if filter_by_score:
                        # update EMA score (only for self-training)
                        self.unknown_score_ema[lbl] = self.unknown_score_ema[lbl] * self.mom + (1.0 - self.mom) * pred_scores[i]
    
                        thresh_names = ['unkthresh', 'ema', 'min_score']
                        threshs = np.array([unknown_threshs[lbl], self.unknown_score_ema[lbl], self.min_score])
                        thresh_idx = np.argmax(threshs)

                        score_thresh = threshs[thresh_idx]
                        # score_thresh = max(unknown_threshs[lbl], self.unknown_score_ema[lbl], self.min_score)
                        # score_thresh = max(self.unknown_score_ema[lbl], self.min_score)

                        pred_mask[i] &= pred_scores[i] >= score_thresh

                        if pred_mask[i]:
                            num_after[lbl] += 1

            # filter out known boxes
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]
            pred_labels = pred_labels[pred_mask]

        if pred_boxes.shape[0] == 0:
            return np.zeros((0, 8)), np.zeros((0))

        # now turn into (N, 8) gt_boxes
        # pseudo_dim = pred_boxes.shape[-1
        box_dim = 7
        pseudo_boxes = np.zeros((pred_boxes.shape[0], box_dim+1), dtype=np.float32)

        pseudo_boxes[:pred_boxes.shape[0], :box_dim] = pred_boxes[:, :box_dim] # remove velocity etc
        pseudo_boxes[:pred_boxes.shape[0], -1] = pred_labels

        assert pseudo_boxes.shape[0] == pred_scores.shape[0], f'boxes and scores should match on dim0! boxes={pseudo_boxes.shape} scores={pred_scores.shape}'

        return pseudo_boxes, pred_scores

    def load_frustum_pseudos(self, batch_dict: Dict) -> Dict:
        # frustum pseudos (no filtering, score is based on image detector)
        pseudo_boxes, pseudo_scores = self.load_pseudos(batch_dict, filter_by_score=False) # no filtering for frustum pseudos

        batch_dict['pseudo_boxes'] = pseudo_boxes
        batch_dict['pseudo_scores'] = pseudo_scores
        batch_dict['pseudo_samples_mask'] = np.zeros((len(pseudo_boxes, )), dtype=bool)

        self.copy_boxes = pseudo_boxes.copy()
        self.copy_scores = pseudo_scores.copy()

        return batch_dict

    # def __call__(self, batch_dict: Dict) -> Dict:
    def load_selftrain_pseudos(self, batch_dict: Dict) -> Dict:
        """
        Args:
            batch_dict:
                gt_boxes: gt_boxes of known categories
        Returns:
            batch_dict:
                gt_boxes: gt_boxes of knowns, with pseudo labels of unknowns
        """
        if not self.training: 
            print('not training??')
            return batch_dict # do nothing

        if 'pseudo_boxes' in batch_dict: # if frustum_pseudos aug added
            pseudo_boxes, pseudo_scores = batch_dict.pop('pseudo_boxes', np.zeros((0, 8))), \
                batch_dict.pop('pseudo_scores', np.zeros((0)))
        else:
            pseudo_boxes, pseudo_scores = np.zeros((0, 8)), np.zeros((0))

        # print('ema scores', self.unknown_score_ema.items())

        # pseudos from self-training
        last_round_boxes, last_round_scores = self.load_pseudos(batch_dict, folder=self.self_training_folder, record_missing=False)
        num_st = len(last_round_boxes)
        num_frst = len(pseudo_boxes)

        # if self-training pseudo labels exist
        if num_st > 0:
            pseudo_boxes = np.concatenate([pseudo_boxes, last_round_boxes], axis=0)
            pseudo_scores = np.concatenate([pseudo_scores, last_round_scores], axis=0)

        # need torch for nms
        pseudo_boxes_tensor = torch.tensor(pseudo_boxes, dtype=torch.float32)
        pseudo_scores_tensor = torch.tensor(pseudo_scores, dtype=torch.float32)

        pseudo_types_tensor = torch.ones((len(pseudo_boxes, )), dtype=torch.long)
        pseudo_types_tensor[:num_frst] = 0

        # if num_frst > 0 and num_st > 0:
        #     pseudo_classes = pseudo_boxes_tensor[:, -1].long()
        #     print('pseudo_classes', pseudo_classes)
        #     ious = iou3d_nms_utils.boxes_bev_iou_cpu(pseudo_boxes_tensor[:num_frst, :7], pseudo_boxes_tensor[num_frst:(num_frst + num_st), :7])
        #     frst_classes = pseudo_classes[:num_frst]
        #     st_classes = pseudo_classes[num_frst:(num_frst + num_st)]

        #     # print('pseudo max iou', ious.max(dim=1).values)
        #     ious_max = ious.max(dim=1)

        #     for frst_idx, (st_idx, iou) in enumerate(zip(ious_max.indices, ious_max.values)):
        #         frst_cls = frst_classes[frst_idx].item()
        #         st_cls = st_classes[st_idx].item()

        #         # print('frst, st, iou', frst_cls, st_cls, iou)
        #         if frst_cls == st_cls and iou > 0.1:
        #             print(f'cls match, iou={iou}')
        #             print(pseudo_boxes_tensor[frst_idx], pseudo_boxes_tensor[(num_frst+st_idx)])
        #             print(pseudo_scores_tensor[frst_idx], pseudo_scores_tensor[(num_frst+st_idx)])

            
        # do nms with self-training and frustum pseudos
        indices = bev_nms_cpu(pseudo_boxes_tensor[:, :7], pseudo_scores_tensor, thresh=0.1)
        nms_removed = pseudo_boxes_tensor.shape[0] - indices.shape[0]
        # print('pseudo nms removed', nms_removed, '/', pseudo_boxes_tensor.shape[0])

        pseudo_boxes_tensor = pseudo_boxes_tensor[indices]#.numpy()
        pseudo_scores_tensor = pseudo_scores_tensor[indices]
        pseudo_types_tensor = pseudo_types_tensor[indices]

        # pseudo_removed = (num_pseudo - (pseudo_types_tensor == 0).sum())
        # st_removed = (num_last_round - (pseudo_types_tensor == 1).sum())


        # find pseudos that don't overlap with GT
        # pseudo_boxes_tensor = torch.tensor(pseudo_boxes.copy(), dtype=torch.float32)
        gt_boxes_tensor = torch.tensor(batch_dict['gt_boxes'].copy(), dtype=torch.float32)

        # add ego to gt to remove any pseudos that overlap
        gt_boxes_tensor = torch.cat((gt_boxes_tensor[:, :7], self.ego_vehicle.to(gt_boxes_tensor.device)), dim=0)

        # remove those that have any overlap with GT
        if gt_boxes_tensor.numel() > 0 and pseudo_boxes_tensor.numel() > 0:
            ious = iou3d_nms_utils.boxes_bev_iou_cpu(pseudo_boxes_tensor[..., :7], gt_boxes_tensor[..., :7]) # use this cpu method 
            
            ious = ious.max(dim=1).values
            ious_mask = (ious <= self.pseudo_nms_thresh)

            # filter those with overlap
            pseudo_boxes = pseudo_boxes_tensor[ious_mask].numpy()
            pseudo_scores = pseudo_scores_tensor[ious_mask].numpy()
            pseudo_types = pseudo_types_tensor[ious_mask].numpy()

        else:
            pseudo_boxes = pseudo_boxes_tensor.numpy()
            pseudo_scores = pseudo_scores_tensor.numpy()
            pseudo_types = pseudo_types_tensor.numpy()

        # remove boxes that are empty
        pseudo_boxes, empty_mask = remove_empty(pseudo_boxes)
        # sample_box_mask = sample_box_mask[empty_mask]
        pseudo_scores = pseudo_scores[empty_mask]
        pseudo_types = pseudo_types[empty_mask]

        # assert sample_box_mask.shape[0] == pseudo_boxes.shape[0], f'pseudo box dim0 should match sample_box_mask dim0, box={pseudo_boxes.shape}, mask={sample_box_mask.shape}'
        assert pseudo_scores.shape[0] == pseudo_boxes.shape[0], f'boxes != scores, box={pseudo_boxes.shape}, scores={pseudo_scores.shape}'

        batch_dict['pseudo_boxes'] = pseudo_boxes
        batch_dict['pseudo_scores'] = pseudo_scores

        self.pseudo_types = pseudo_types
        if self.copy_st_only:
            st_mask = (pseudo_types == 1)

            self.copy_boxes = pseudo_boxes[st_mask].copy()
            self.copy_scores = pseudo_scores[st_mask].copy()
        else:
            self.copy_boxes = pseudo_boxes.copy()
            self.copy_scores = pseudo_scores.copy()

        batch_dict['pseudo_samples_mask'] = np.zeros((len(pseudo_boxes), ), dtype=bool)

        return batch_dict
    
    def copy_and_paste(self, batch_dict: Dict) -> Dict:
        # pseudo sampling to get more boxes for unknowns
        pseudo_boxes, sample_box_mask = self.sampler(batch_dict, self.copy_boxes, self.copy_scores, batch_dict['gt_boxes'], fix_cp=self.fix_cp)

        # remove boxes that are empty
        pseudo_boxes, empty_mask = remove_empty(pseudo_boxes)
        sample_box_mask = sample_box_mask[empty_mask]

        if self.copy_st_only:
            # add back the
            non_st_mask = (self.pseudo_types == 0)
            frst_boxes = batch_dict['pseudo_boxes'][non_st_mask]

            pseudo_boxes = np.concatenate([pseudo_boxes, frst_boxes], axis=0)
            sample_box_mask = np.concatenate([sample_box_mask, np.zeros((len(frst_boxes), ), dtype=bool)], axis=0)

        assert sample_box_mask.shape[0] == pseudo_boxes.shape[0], f'pseudo box dim0 should match sample_box_mask dim0, box={pseudo_boxes.shape}, mask={sample_box_mask.shape}'

        batch_dict.pop('pseudo_scores')

        batch_dict['pseudo_boxes'] = pseudo_boxes
        batch_dict['pseudo_samples_mask'] = sample_box_mask

        return batch_dict
