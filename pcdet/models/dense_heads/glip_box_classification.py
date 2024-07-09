import argparse
import glob
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torchvision.ops import box_iou

from pcdet.utils.box_utils import boxes_to_corners_3d

from pcdet.models.preprocessed_detector import PreprocessedDetector, PreprocessedGLIP

class GLIPBoxClassification(nn.Module):
    
    def __init__(self, image_size=[900, 1600], all_class_names=None, model_cfg=None) -> None:
        super().__init__()

        self.model_cfg = model_cfg if model_cfg is not None else dict()

        self.image_order = [0, 1, 2, 3, 4, 5] # for plotting
        self.image_size = image_size

        if all_class_names is None:
            self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        else:
            self.all_class_names = all_class_names

        preds_path = self.model_cfg.get('PREDS_PATH', '/home/uqdetche/GLIP/jsons/OWL_')
        self.box_fmt = self.model_cfg.get('BOX_FORMAT', 'xyxy')


        print('preds_path', preds_path)
        if 'PreprocessedGLIP' in preds_path:
            self.detector = PreprocessedGLIP(class_names=self.all_class_names)
        else:
            camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
            preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

            preds_paths = model_cfg.get('PREDS_PATHS', preds_paths)
            print('preds_paths', preds_paths)
            self.detector = PreprocessedDetector(preds_paths, class_names=all_class_names)

        # self.detector = PreprocessedGLIP(class_names=self.all_class_names)
        # self.detector = PreprocessedDetector(cam_jsons=['../data/training_pred/glip_train_preds.json'], class_names=self.all_class_names)

    def forward(self, batch_dict, pred_dicts, batch_idx=0):
        """
        Args:
            batch_dict:
                batch information containing, e.g. images, camera_intrinsics etc keyss

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        batch_size = batch_dict['batch_size']

        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        # load glip training predictions
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.detector(batch_dict)


        # for each point cloud in the batch
        for b in range(batch_size):
            cur_pred_dict = pred_dicts[b]

            cur_boxes = cur_pred_dict['pred_boxes']
            N = cur_boxes.shape[0]

            if N == 0:
                # print('no boxes for clip!')
                continue

            box_probs = torch.zeros((N, 6, 10), device='cuda', dtype=torch.half) # 6 cameras, 10 classes
            box_cam_mask = torch.zeros((N, 6), device='cuda')

            # box number
            cur_idx = torch.arange(0, N).reshape(N, 1).repeat(1, 8).reshape(N*8)
            corners = boxes_to_corners_3d(cur_boxes)
            corners = corners.reshape(-1, 3)

            cur_coords = corners
            # cur_images = images[b]

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
            # cur_coords = cur_coords[..., [1, 0]]

            # for each picture in a different view
            for c in self.image_order:

                detector_mask = (det_batch_idx == b) & (det_cam_idx == c)
                glip_bbox_coor, glip_bbox_label, glip_bbox_score = det_boxes[detector_mask], \
                      det_labels[detector_mask], det_scores[detector_mask]
                
                # check if there are no GLIP boxes
                if glip_bbox_coor.numel() == 0:
                    continue

                if self.box_fmt != 'xyxy':
                    glip_bbox_coor[..., 2:] += glip_bbox_coor[..., 0:2]

                # print('boxes', glip_bbox_coor, 'labels', glip_bbox_label, glip_bbox_score)
                # print('boxes', glip_bbox_coor.shape, 'labels', glip_bbox_label.shape, glip_bbox_score.shape)
                # glip_bbox_coor = self.glip_bboxes[img_id].bbox
                # glip_bbox_score = self.glip_bboxes[img_id].extra_fields['scores']
                # glip_bbox_label = self.glip_bboxes[img_id].extra_fields['labels']
                # torchvision.utils.save_image(images[b][c], 'visualization/{}_{}_{}_vis.png'.format(b, c, img_id))


                all_coords = cur_coords[c, :].long().cpu()

                projected_bboxes = []
                sampled_idx = []

                # for each predicted 3d bbox, check if valid in the current c view
                for idx in range(N):
                    coord_mask = (cur_idx == idx)

                    # change to new way
                    # coords2d = [[x[1], x[0]] for x in all_coords[coord_mask]]
                    # box = post_process_coords(coords2d)
                    print('all_coords', all_coords.shape)
                    image_pos = all_coords[coord_mask, :2]

                    # clamp to image dimensions
                    image_pos[..., 0] = torch.clamp(image_pos[..., 0], 0, self.image_size[1])
                    image_pos[..., 1] = torch.clamp(image_pos[..., 1], 0, self.image_size[0])

                    # print('image_pos', image_pos.shape)
                    # get bbox
                    xy1 = image_pos.min(dim=0).values
                    xy2 = image_pos.max(dim=0).values
                    proj_box = torch.zeros((4))
                    proj_box[0:2] = xy1
                    proj_box[2:] = xy2

                    wh = (xy2 - xy1)

                    # check projected box has width and height
                    if not (wh > 0).all():
                        continue

                    # this box occurs on this camera
                    box_cam_mask[idx, c] = 1

                    # save coordi
                    projected_bboxes.append(proj_box)
                    sampled_idx.append(idx)

                if len(projected_bboxes) == 0:
                    continue

                projected_bboxes = torch.stack(projected_bboxes, dim=0)
                sampled_idx = torch.tensor(sampled_idx)
            
                print('projected_bboxes', projected_bboxes.shape)

                # leverage glip prediction to generate new logits
                with torch.no_grad():
                    iou_matrix = box_iou(glip_bbox_coor.cuda(), projected_bboxes.cuda()).float()
                    label_matrix = F.one_hot(glip_bbox_label - 1, num_classes=len(self.all_class_names)).cuda().float() * glip_bbox_score.unsqueeze(-1).cuda()
                    print('iou_matrix', iou_matrix.shape)
                    print('label_matrix', label_matrix.shape)
                    probs = iou_matrix.T @ label_matrix 
                    box_probs[sampled_idx, c] = probs.to(box_probs.dtype)

            # mean over the camera images (that this box actually showed in)
            box_probs_mean = box_probs.sum(dim=1) / (1e-5 + box_cam_mask.sum(dim=-1).unsqueeze(1))

            # pred_labels = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)
            # pred_labels = pred_labels.flatten().cpu() + 1  # 0 -> bg, 1 -> car, ...

            # pred_labels = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)
            probs_max = torch.max(box_probs_mean.cpu(), dim=-1)
            pred_scores = probs_max.values
            pred_labels = probs_max.indices.flatten()

            pred_scores = torch.nan_to_num(pred_scores, nan=0.0)

            # pred_dicts[b]['pred_labels'] = pred_labels + 1 # 0 -> bg, 1 -> car, ...
            pred_dicts[b]['pred_labels'] = pred_labels + 1 # 0 -> bg, 1 -> car, ...
            pred_dicts[b]['pred_scores'] = pred_scores # replace with clip scores

            # pred_scores = torch.max(box_probs_mean.cpu(), dim=-1, keepdim=True)[0]
            
            # # consider remove low quality ones [TODO: Centerpoint's labels have been mapped already]
            # orig_pred_labels = self.known_to_full_mapping[(pred_dicts[b]['pred_labels'] - 1).long()]

            # # orig_pred_labels = pred_dicts[b]['pred_labels'].cpu() 
            # orig_pred_labels = orig_pred_labels.to(pred_labels.dtype)

            # if self.conf_fusion == 'thresh_lvm':
            #     pred_labels[pred_scores.squeeze() < self.lvm_conf_threshold] = orig_pred_labels[pred_scores.squeeze() < self.lvm_conf_threshold]
            #     fused_labels = pred_labels
            # elif self.conf_fusion == 'thresh_orig':
            #     orig_pred_scores = pred_dicts[b]['pred_scores'].cpu()
            #     orig_pred_labels[orig_pred_scores.squeeze() < self.orig_conf_threshold] = pred_labels[orig_pred_scores.squeeze() < self.orig_conf_threshold]
            #     fused_labels = orig_pred_labels
            # elif self.conf_fusion == 'thresh_unk':
            #     for i in range(pred_labels.shape[0]):
            #         if pred_labels[i] not in self.known_to_full_mapping.cpu() and pred_scores[i] > self.lvm_conf_threshold:
            #             orig_pred_labels[i] = pred_labels[i]
            #     fused_labels = orig_pred_labels
            # else:
            #     raise NotImplementedError
            
            # # replace
            # pred_dicts[b]['pred_labels'] = fused_labels 
            # new_bbox_names = np.array(self.all_class_names)[fused_labels - 1]
            # if isinstance(new_bbox_names, str):
            #     new_bbox_names = np.array([new_bbox_names])

            # pred_dicts[b]['name'] = new_bbox_names
            # pred_dicts[b]['glip_score'] = pred_scores
            
        return pred_dicts

# class GLIPBoxClassification(nn.Module):
#     def __init__(self, image_size=[900, 1600], all_class_names=None, model_cfg=None) -> None:
#         super().__init__()

#         self.model_cfg = model_cfg if model_cfg is not None else dict()

#         self.image_order = [0, 1, 2, 3, 4, 5] # for plotting
#         self.image_size = image_size

#         if all_class_names is None:
#             self.all_class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
#                 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
#         else:
#             self.all_class_names = all_class_names

#         preds_path = self.model_cfg.get('PREDS_PATH', '/home/uqdetche/GLIP/jsons/OWL_')
#         self.box_fmt = self.model_cfg.get('BOX_FORMAT', 'xyxy')


#         print('preds_path', preds_path)
#         if 'PreprocessedGLIP' in preds_path:
#             self.detector = PreprocessedGLIP(class_names=self.all_class_names)
#         else:
#             camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
#             preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

#             preds_paths = model_cfg.get('PREDS_PATHS', preds_paths)
#             print('preds_paths', preds_paths)
#             self.detector = PreprocessedDetector(preds_paths, class_names=all_class_names)

#         # self.detector = PreprocessedGLIP(class_names=self.all_class_names)
#         # self.detector = PreprocessedDetector(cam_jsons=['../data/training_pred/glip_train_preds.json'], class_names=self.all_class_names)

#         self.pred_jsons = pred_jsons
#         self.class_names = all_class_names
#         # self.detector = PreprocessedDetector(pred_jsons, class_names=class_names)
#         self.min_iou = min_iou
#         self.image_size = image_size

#     def clip_coords(self, masked_coords):
#         # self.clip_to_image(masked_coords)
#         x1, x2 = masked_coords[..., 1].min(), masked_coords[..., 1].max()
#         y1, y2 = masked_coords[..., 0].min(), masked_coords[..., 0].max()

#         x1 = torch.clamp(x1, 0, self.image_size[1])
#         x2 = torch.clamp(x2, 0, self.image_size[1])
#         y1 = torch.clamp(y1, 0, self.image_size[0])
#         y2 = torch.clamp(y2, 0, self.image_size[0])

#         return x1, y1, x2, y2


#     def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
#         cur_coords = points.clone()

#         # camera_intrinsics = batch_dict['camera_intrinsics']
#         # camera2lidar = batch_dict['camera2lidar']
#         img_aug_matrix = batch_dict['img_aug_matrix']
#         lidar_aug_matrix = batch_dict['lidar_aug_matrix']
#         lidar2image = batch_dict['lidar2image']

#         cur_img_aug_matrix = img_aug_matrix[batch_idx, [cam_idx]]
#         cur_lidar_aug_matrix = lidar_aug_matrix[batch_idx]
#         cur_lidar2image = lidar2image[batch_idx, [cam_idx]]

#         # inverse aug
#         cur_coords -= cur_lidar_aug_matrix[:3, 3]
#         cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
#             cur_coords.transpose(1, 0)
#         )
#         # lidar2image
#         cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
#         cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
#         # get 2d coords
#         cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :].clone(), 1e-5, 1e5)
#         cur_coords[:, :2, :] /= cur_coords[:, 2:3, :].clone()

#         # do image aug
#         cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
#         cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
#         cur_coords = cur_coords[:, :3, :].transpose(1, 2)

#         # normalize coords for grid sample
#         cur_coords = cur_coords[..., [1, 0, 2]]

#         # filter points outside of images
#         # on_img = (
#         #     (cur_coords[..., 1] < self.image_size[0])
#         #     & (cur_coords[..., 1] >= 0)
#         #     & (cur_coords[..., 0] < self.image_size[1])
#         #     & (cur_coords[..., 0] >= 0)
#         # )

#         return cur_coords

#     def forward(self, batch_dict, pred_dicts, relabel=True):
#         """
#         Args:
#             batch_dict:
#                 batch information containing, e.g. images, camera_intrinsics etc keyss

#         Returns:
#             batch_dict:
#                 spatial_features_img (tensor): bev features from image modality
#         """

#         batch_size = batch_dict['batch_size']

#         images = batch_dict['camera_imgs']

#         camera_intrinsics = batch_dict['camera_intrinsics']
#         camera2lidar = batch_dict['camera2lidar']
#         img_aug_matrix = batch_dict['img_aug_matrix']
#         lidar_aug_matrix = batch_dict['lidar_aug_matrix']
#         lidar2image = batch_dict['lidar2image']


#         det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.detector(batch_dict)

#         for b in range(batch_size):
#             cur_pred_dict = pred_dicts[b]

#             pred_boxes = cur_pred_dict['pred_boxes']
#             N = pred_boxes.shape[0]

#             detector_batch_mask = (det_batch_idx == b)
#             cur_boxes, cur_det_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]


#             if N == 0:
#                 print('no boxes for clip!')
#                 print(cur_pred_dict)
#                 continue

#             proj_boxes = torch.zeros((N, 4), dtype=torch.float, device='cuda')
#             box_probs = torch.zeros((N, 6, 10), device='cuda', dtype=torch.half) # 6 cameras, 10 classes
#             box_cam_mask = torch.zeros((N, 6), device='cuda')

#             # box number
#             cur_idx = torch.arange(0, N).reshape(N, 1).repeat(1, 8).reshape(N*8)
#             corners = boxes_to_corners_3d(pred_boxes)
#             corners = corners.reshape(-1, 3)


#             for c in range(6):
#                 cur_coords = self.project_to_camera(batch_dict, corners, b, c)
#                 cur_cam_mask = (cur_cam_idx == c)
#                 cam_boxes, cam_labels, cam_scores = cur_boxes[cur_cam_mask], cur_det_labels[cur_cam_mask], cur_scores[cur_cam_mask]

#                 all_coords = cur_coords[0, ..., :2].long()

#                 for idx in range(N):
#                     coord_mask = (cur_idx == idx)
#                     box_cam_coords = all_coords[coord_mask]

#                     x1, x2 = box_cam_coords[..., 1].min(), box_cam_coords[..., 1].max()
#                     y1, y2 = box_cam_coords[..., 0].min(), box_cam_coords[..., 0].max()

#                     proj_boxes[idx, 0] = torch.clamp(x1, 0, self.image_size[1])
#                     proj_boxes[idx, 1] = torch.clamp(y1, 0, self.image_size[0])
#                     proj_boxes[idx, 2] = torch.clamp(x2, 0, self.image_size[1])
#                     proj_boxes[idx, 3] = torch.clamp(y2, 0, self.image_size[0])

#                 cam_boxes = cam_boxes.to(proj_boxes.device)
#                 ious = box_iou(proj_boxes, cam_boxes)
#                 print('ious', ious.shape)

#                 ious_max = ious.max(dim=1)
#                 print('ious', ious_max.values)

#                 for idx in range(N):
#                     cam_matched_box = ious_max.indices[idx]
#                     cam_label = cam_labels[cam_matched_box] - 1
#                     match_iou = ious_max.values[idx]

#                     if match_iou >= self.min_iou: 
#                         box_probs[idx, c, cam_label] = match_iou
#                         box_cam_mask[idx, c] += 1


#             # mean over the camera images (that this box actually showed in)
#             box_probs_mean = box_probs.sum(dim=1) / box_cam_mask.sum(dim=-1).unsqueeze(1).clamp(min=1)
#             print('box_probs_mean', box_probs_mean)

#             # pred_labels = torch.argmax(box_probs_mean.cpu(), dim=-1, keepdim=True)
#             probs_max = torch.max(box_probs_mean.cpu(), dim=-1)
#             pred_scores = probs_max.values
#             pred_labels = probs_max.indices.flatten() + 1  # 0 -> bg, 1 -> car, ...

#             # background
#             pred_labels[pred_scores < 0.01] = 0

#             print('pred_scores', pred_scores)

            
#             if relabel:
#                 pred_dicts[b]['pred_labels'] = pred_labels
#                 pred_dicts[b]['pred_scores'] = pred_scores # replace with clip scores

#         return pred_dicts