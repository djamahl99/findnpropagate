import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
import os

class PointHeadBoxWPseudos(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        num_class = 10
        self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.known_class_names = model_cfg.KNOWN_CLASS_NAMES

        self.is_known = {i: (cls in self.known_class_names) for i, cls in enumerate(self.all_class_names)}

        # gt_labels are 1 indexed
        self.gt_known_to_full_labels = {(i + 1): (j + 1) for i, known_name in enumerate(self.known_class_names) for j, all_name in enumerate(self.all_class_names) if known_name == all_name}
        self.full_labels_to_gt_known = {v: k for k, v in self.gt_known_to_full_labels.items()}

        for i, cls in enumerate(self.all_class_names):
            mp_str = ""
            if (i + 1) in self.full_labels_to_gt_known:
                mp_str = f"no. {self.full_labels_to_gt_known[i+1]} => {i+1} full labels"
                mp_str += f", {self.all_class_names[i]} => {self.known_class_names[self.full_labels_to_gt_known[i + 1] - 1]}"
            print(f"{cls}: {'Known' if self.is_known[i] else 'Unknown'}", mp_str)

        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )

        self.pseudo_folder = 'pseudo_labels/frustum_proposals/'

    def relabel_gt_boxes(self, gt_boxes: torch.tensor) -> torch.tensor:
        """
        As the idx changes when remove classes from the class list, need to 
        remap back to original idx for original 10 classes.

        Args:
            gt_boxes (torch.tensor): gt boxes

        Returns:
            torch.tensor: gt_boxes relabeled to full class idx
        """
        for b in range(gt_boxes.shape[0]):
            for i in range(gt_boxes.shape[1]):
                curr_label = int(gt_boxes[b, i, -1].item())
                if curr_label in self.gt_known_to_full_labels:
                    # only modify if not the non-match id
                    gt_boxes[b, i, -1] = self.gt_known_to_full_labels[curr_label]

        return gt_boxes

    def load_pseudos(self, batch_dict, unknowns_only=True) -> torch.tensor:
        """Load pseudo labels

        Args:
            batch_dict (dict): batch data
            unknowns_only (bool): only return pseudo boxes for unknowns
            
        Returns:
            torch.tensor: pseudo boxes (B, M, 8)
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
                print('psuedos dont exist!')
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
    
    def combine_gt_with_pseudos(self, gt_boxes: torch.tensor, pseudo_boxes: torch.tensor) -> torch.tensor:
        """Combine gt boxes for known classes with pseudo boxes for unknowns

        Args:
            gt_boxes (torch.tensor): gt boxes (B, N, 8) from dataloader, relabeled by relabel_gt_boxes
            pseudo_boxes (torch.tensor): pseudo boxes (B, M, 8)

        Returns:
            torch.tensor: combined boxes (B, N+M, 8)
        """
        def valid_boxes(cur_gt_bboxes_3d: torch.tensor) -> list:
            valid_idx = []
            # filter empty boxes
            for i in range(len(cur_gt_bboxes_3d)):
                if cur_gt_bboxes_3d[i][3] > 0 and cur_gt_bboxes_3d[i][4] > 0: # size > 0
                    valid_idx.append(i)

            return cur_gt_bboxes_3d[valid_idx]

        MN = 0
        MN_ = gt_boxes.shape[1] + pseudo_boxes.shape[1] # adds redundant padding
        ret_boxes = torch.zeros((gt_boxes.shape[0], MN_, 8), device=gt_boxes.device)

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
            ret_boxes[b, num_gt:num] = cur_pseudo_boxes

        ret_boxes = ret_boxes[:, :MN].contiguous() # remove redundant padding

        return ret_boxes

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']

        # relabel gt to full class labels and then combine with pseudo labels
        gt_boxes = self.relabel_gt_boxes(gt_boxes)
        pseudo_boxes = self.load_pseudos(input_dict)
        gt_boxes = self.combine_gt_with_pseudos(gt_boxes, pseudo_boxes)

        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)

        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            print('point_cls_preds', point_cls_preds.shape, point_cls_preds[0])
            print('point_box_preds', point_box_preds.shape, point_box_preds[0])
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
