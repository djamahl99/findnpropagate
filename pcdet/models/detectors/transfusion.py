import torch
from .detector3d_template import Detector3DTemplate


class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}

        if post_process_cfg.get('RELABEL_NUSC', False):
            nusc_10_class = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            kitti_classes = ['Car', 'Tram', 'Truck', 'Van', 'Person_sitting', 'Cyclist', 'Pedestrian']

            mapping = {
                'car': 'Car',
                'truck': 'Truck',
                # 'construction_vehicle': 'Misc',
                # 'bus': 'Misc',
                # 'trailer': 'Truck',
                # 'barrier': 'Misc',
                # 'motorcycle': 'Misc',
                'bicycle': 'Cyclist',
                'pedestrian': 'Pedestrian',
                # 'traffic_cone': 'Misc'
            }

            nusc_to_kitti = {(i + 1): (j + 1) for i, nusc in enumerate(nusc_10_class) for j, kitti in enumerate(kitti_classes) if nusc in mapping and mapping[nusc] == kitti}

        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            if post_process_cfg.get('RELABEL_NUSC', False):
                pred_labels = final_pred_dict[index]['pred_labels']

                pred_mask = torch.ones_like(pred_labels, dtype=torch.bool)

                for i, lbl in enumerate(pred_labels):
                    if lbl.item() in nusc_to_kitti:
                        final_pred_dict[index]['pred_labels'][i] = nusc_to_kitti[lbl.item()]
                    else:
                        # remove predictions for classes not in kitti
                        pred_mask[i] = False

                for k in ['pred_labels', 'pred_boxes', 'pred_scores']:
                    final_pred_dict[index][k] = final_pred_dict[index][k][pred_mask]

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        if self.vlm is not None:
            pred_dicts = self.vlm(batch_dict, final_pred_dict)

        return final_pred_dict, recall_dict
