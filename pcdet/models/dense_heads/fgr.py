import copy
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import common_utils

from pcdet.utils.box_utils import boxes_to_corners_3d
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils
from torchvision.ops import box_iou, batched_nms, nms
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from ..preprocessed_detector import PreprocessedDetector, PreprocessedGLIP

from matplotlib import pyplot as plt

from ..model_utils import model_nms_utils
import time
import cv2
from .fgr_utils import *

class FGR(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True
    ):
        super(FGR, self).__init__()

        self.model_cfg = model_cfg

        self.nms_2d = 0.4
        self.score_thr = 0.1
        self.image_order = [2, 0, 1, 5, 3, 4]
        self.image_size = [900, 1600]
        self.image_shape_xy = [1600, 900]
        self.point_cloud_range = point_cloud_range

        self.thresh_ransac = model_cfg.THRESH_RANSAC
        self.thresh_seg_max = model_cfg.THRESH_SEG_MAX
        self.ratio = model_cfg.REGION_GROWTH_RATIO

        self.box_fmt = model_cfg.get('BOX_FORMAT', 'xyxy')
        preds_path = model_cfg.get('PREDS_PATH', '/home/uqdetche/GLIP/jsons/OWL_')

        if 'PreprocessedGLIP' in preds_path:
            self.image_detector = PreprocessedGLIP(class_names=class_names)
        else:
            camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
            preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

            preds_paths = model_cfg.get('PREDS_PATHS', preds_paths)
            self.image_detector = PreprocessedDetector(preds_paths, class_names=class_names)

    def get_proposals(self, batch_dict):
        return self.region_growth(batch_dict)

    def region_growth(self, batch_dict):
        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        # lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        # 2d multiview detections loaded (loaded from coco jsons)
        det_boxes, det_labels, det_scores, det_batch_idx, det_cam_idx = self.image_detector(batch_dict)
        det_idx = torch.arange(det_boxes.shape[0])

        full_results = []
        full_labels = []
        full_batch_idx = []
        full_scores = []

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]
            # plane_mask = fit_plane(cur_points)
            # # print('num in plane', plane_mask.shape, plane_mask.sum(), plane_mask.sum() / cur_points.shape[0])
            # non_ground = torch.bitwise_not(plane_mask)
            # cur_points = cur_points[non_ground]
            foreground_pts = cur_points

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]


            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                if cam_boxes.shape[0] > 0:
                    selected = batched_nms(cam_boxes, cam_scores, cam_labels, self.nms_2d)
                    cam_boxes, cam_labels, cam_scores = cam_boxes[selected], cam_labels[selected], cam_scores[selected]

                cam_box_idx = torch.arange(cam_boxes.shape[0])

                # project points onto the camera
                cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                cam_points = cam_points[cam_mask]

                cam_points_3d = cur_points[cam_mask[0]]

                # pc_all = cam_points.reshape(-1, 3).detach().cpu().numpy()
                pc_all = cam_points_3d.reshape(-1, 3).detach().cpu().numpy()
                
                # kitti camera coordinates are switched for the camera frame
                # x y z -> y z x
                # z -> y, y-> x, x -> z
                pc_all = pc_all[:, [1, 2, 0]]
                cam_points_3d = cam_points_3d[:, [1, 2, 0]]

                # [2, 0, 1]

                object_filter_all = torch.zeros(cam_points.shape[0], dtype=torch.bool, device=cam_points.device)
                print('object_filter_all', object_filter_all.dtype)
                
                camera_mask_ground_all = []
                camera_ground_sample_points = []

                valid_list = []
                index_list = []
                z_list = []
                object_filter_list = [None for _ in cam_box_idx]


                # cam_points_3d = cur_points[cam_mask]

                # FGR calculate ground
                mask_ground_all, ground_sample_points = calculate_ground(pc_all, 0.2, back_cut=False)
                mask_ground_all = mask_ground_all.astype('bool')
                # print('mask_ground_all', mask_ground_all.sum(), mask_ground_all.shape)
                # print('ground points', pc_all[mask_ground_all])
                # print('ground_sample_points', ground_sample_points)
                # exit()

                for i, box, label, score in zip(cam_box_idx, cam_boxes, cam_labels, cam_scores):
                    if score < self.score_thr:
                        continue

                    flag = 1
                    
                    if self.box_fmt == 'xyxy':
                        x1, y1, x2, y2 = box.cpu()
                    else:
                        box[..., 2:] += box[..., 0:2]
                        x1, y1, x2, y2 = box

                    on_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    # get 3d points in the 2d box
                    pc = cam_points_3d[on_box].detach().cpu().numpy()

                    # object_filter_all[cam_mask[0]] = torch.bitwise_or(object_filter_all[cam_mask[0]], on_box)
                    object_filter_all[on_box] = True

                    if on_box.sum() > 0:
                        assert object_filter_all.sum() > 0

                    # object filter in shape of full points
                    # object_filter = cam_mask.clone().reshape(-1)
                    # object_filter[cam_mask[0]] = False
                    # object_filter[cam_mask[0]] = on_box.reshape(-1)

                    object_filter_list[i] = on_box.detach().cpu().numpy()

                    if on_box.sum() < 30:
                        flag = 0

                    if flag == 1:
                        valid_list.append(i)

                    z_list.append(np.median(pc[:, 2]))
                    index_list.append(i)

                object_filter_all = object_filter_all.detach().cpu().numpy()

                sort = np.argsort(np.array(z_list))
                object_list = list(np.array(index_list)[sort])

                mask_object = np.ones((pc_all.shape[0]))

                # [add] dict to be saved
                dic = {
                    'shape': self.image_shape_xy,
                    'ground_sample': ground_sample_points,
                    'sample': {}
                }

                for i in object_list:
                    result = np.zeros((7, 2))
                    count = 0
                    mask_seg_list = []

                    object_filter = object_filter_list[i]

                    assert object_filter is not None

                    # filter_z = pc_all[:, 2] > 0
                    # filter_z = pc_all[:, 2] != 0 # not important (was for kitti)
                    # filter_z = np.linalg.norm(pc_all, axis=1) > 0
                    mask_search = mask_ground_all * object_filter_all * mask_object #* filter_z
                    print('mask_ground_all', mask_ground_all.sum(), 'object_filter_all', object_filter_all.sum(), 'mask_object', mask_object.sum())

                    print('mask_search', mask_search.sum())
                    
                    if mask_search.sum() == 0:
                        continue

                    for j in range(self.thresh_seg_max):
                        thresh = (j + 1) * 0.1
                        # _, object_filter = kitti_utils_official.get_point_cloud_my_version(
                        #     lidar_path, calib, image_shape_xy, [objects[i].boxes[0]], back_cut=False)



                        # print('mask_search', mask_search.sum())
                        mask_origin = mask_ground_all * object_filter * mask_object #* filter_z
                        mask_seg = region_grow_my_version(pc_all.copy(), mask_search, mask_origin, thresh, self.ratio)
                        if mask_seg.sum() == 0:
                            print('mask seg is zero!')
                            continue

                        if j >= 1:
                            mask_seg_old = mask_seg_list[-1]
                            if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                                count += 1
                        result[count, 0] = j  
                        result[count, 1] = mask_seg.sum()
                        mask_seg_list.append(mask_seg)
                        
                    best_j = result[np.argmax(result[:, 1]), 0]
                    try:
                        mask_seg_best = mask_seg_list[int(best_j)]
                        mask_object *= (1 - mask_seg_best)
                        pc = pc_all[mask_seg_best == 1].copy()
                    except IndexError:
                        print("bad region grow result! deprecated")
                        continue
                    if i not in valid_list:
                        continue

                    if check_truncate(self.image_shape_xy, cam_boxes[i]):

                        mask_origin_new = mask_seg_best
                        mask_search_new = mask_ground_all
                        thresh_new      = (best_j + 1) * 0.1

                        mask_seg_for_truncate = region_grow_my_version(pc_all.copy(),
                                                    mask_search_new,
                                                    mask_origin_new,
                                                    thresh_new,
                                                    ratio=None)
                        pc_truncate = pc_all[mask_seg_for_truncate == 1].copy()
                        dic['sample'][i] = {
                            'truncate': True,
                            'box2d': cam_boxes[i],
                            'label': cam_labels[i],
                            'pc': pc_truncate
                        }

                    else:
                        dic['sample'][i] = {
                            'truncate': False,
                            'box2d': cam_boxes[i],
                            'label': cam_labels[i],
                            'pc': pc
                        }

                # equivalent of generate_result in FGR/detect.py
                cam_results, cam_labels = self.generate_result(batch_dict, dic, batch_idx=b, cam_idx=c)
                # exit()
                for box in cam_results:
                    full_results.append(box)
                full_labels.extend(cam_labels)
                full_batch_idx.extend([b] * len(cam_results))
                full_scores.extend([1.0] * len(cam_results))

        if len(full_results) == 0:
            boxes = torch.zeros((0, 7), device='cuda')
            labels = torch.zeros((0), dtype=torch.long)
            scores = torch.zeros((0))
            batch_idx = torch.zeros((0), dtype=torch.long)

        else:
            boxes = torch.tensor(full_results, device='cuda', dtype=torch.float32)
            print('boxes', boxes.shape)
            boxes = boxes.reshape(-1, 7)
            labels = torch.tensor(full_labels, dtype=torch.long)
            scores = torch.tensor(full_scores)
            batch_idx = torch.tensor(full_batch_idx, dtype=torch.long)

        return boxes, labels, scores, batch_idx

    def generate_result(self, batch_dict, dic, batch_idx: int, cam_idx: int):
        objects = {k: dic['sample'][k] \
                    for k in dic['sample'].keys()}


        if len(list(objects.keys())) == 0:
            print("no valid cars")
            return [], []

        iou_collection = []
        total_object_number = 0
        out_boxes = []
        out_labels = []
        for i in objects.keys():
            pc = dic['sample'][i]['pc']
            label = dic['sample'][i]['label']

            # ignore bad region grow result (with too many points), which may lead to process stuck in deleting points
            if len(pc) > 4000:
                continue

            # for standard data, y_min and y_max is a float number; 
            # while meeting bugs, y_max is an error message while y_min is error code
            img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min = self.Find_2d_box(True, pc, objects[i]['box2d'],
                                                                                   truncate=dic['sample'][i]['truncate'],
                                                                                   sample_points=dic['ground_sample'],
                                                                                   batch_idx=batch_idx,
                                                                                   cam_idx=cam_idx, batch_dict=batch_dict)

            if img_down is None:
                print('img down is none 1', img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min)
                img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min = self.Find_2d_box(False, pc,
                                                                                       objects[i]['box2d'],
                                                                                       truncate=dic['sample'][i]['truncate'],
                                                                                       sample_points=dic['ground_sample'],
                                                                                       batch_idx=batch_idx,
                                                                                    cam_idx=cam_idx, batch_dict=batch_dict)
                if img_down is None:
                    print('img_down is none 2', img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min)
                    continue

            print('loc2', loc2)
            # if loc2 is not None:
                # _, std_iou = iou_3d(objects[i].corners, loc0, loc1, loc2, loc3, y_max=y_max, y_min=y_min)
                # iou_collection.append(std_iou)

            save_p = 'fgr.png'
            # save_p = "fgr.p".format(i)
            print('saving image', save_p)
            cv2.imwrite('fgr_down.png', img_down)
            cv2.imwrite('fgr_side.png', img_side)
            # raise Exception('end!')

            std_box_3d = np.array([[loc0[0], y_max, loc0[1]],
                        [loc1[0], y_max, loc1[1]],
                        [loc2[0], y_max, loc2[1]],
                        [loc3[0], y_max, loc3[1]],

                        [loc0[0], y_min, loc0[1]],
                        [loc1[0], y_min, loc1[1]],
                        [loc2[0], y_min, loc2[1]],
                        [loc3[0], y_min, loc3[1]]])

            std_box_3d = std_box_3d[:, [2, 0, 1]]
            centre = np.mean(std_box_3d, axis=0)

            sides = [
                [loc0, loc1],
                [loc1, loc2],
                [loc2, loc3],
            ]

            angle = np.arctan2((std_box_3d[0, 1] - std_box_3d[1, 1]), (std_box_3d[0, 0] - std_box_3d[1, 0])) + np.pi

            h = y_max - y_min

            RotateMatrix = np.array([[np.cos(-angle), -1 * np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)]])

            xys = (std_box_3d[:, :2] - np.array([centre[0], centre[1]]))
            xys = xys @ RotateMatrix

            x1, x2 = xys[:, 0].min(), xys[:, 0].max()
            y1, y2 = xys[:, 1].min(), xys[:, 1].max()

            # x1, x2 = std_box_3d[:, 0].min(), std_box_3d[:, 0].max()
            # centre = [ , (y_max + y_min)/2.0]
            # h = (y_max - y_min)


            # x1, x2 = std_box_3d[:, 0].min(), std_box_3d[:, 0].max()
            # y1, y2 = std_box_3d[:, 1].min(), std_box_3d[:, 1].max()
            # z1, z2 = std_box_3d[:, 2].min(), std_box_3d[:, 2].max()

            l, w = (x2 - x1), (y2 - y1)

            out_boxes.append([centre[0], centre[1], centre[2], l, w, h, angle])
            out_labels.append(label)

        return out_boxes, out_labels

    def Find_2d_box(self, maximum, KeyPoint_3d, box_2d, 
                    total_pc=None, cluster=None, truncate=False, sample_points=None, batch_idx=0, cam_idx=0, batch_dict=None):
        camera_intrinsics = batch_dict['camera_intrinsics']
        camera2lidar = batch_dict['camera2lidar']
        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        # lidar2image = batch_dict['lidar2image']

        batch_size = batch_dict['batch_size']

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
                    
        # corner is GT corners and not necessary
        detect_config = self.model_cfg

        # ignore less point cases
        if len(KeyPoint_3d) < 10:
            print('not enough key points!')
            return None, None, None, None, None, None, None, None
        
        img = np.zeros((700, 700, 3), 'f4')
        KeyPoint = KeyPoint_3d[:, [0, 2]].copy()

        # left_point  = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[0], 0, 1]).copy().T)[[0, 2]]
        # right_point = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[2], 0, 1]).copy().T)[[0, 2]]
        print('box_2d', box_2d)
        left_point = torch.tensor([box_2d[0], 0, 0],  dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        left_point = self.get_geometry_at_image_coords(left_point, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)

        right_point = torch.tensor([box_2d[2], 0, 0], dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        right_point = self.get_geometry_at_image_coords(right_point, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)

        mat_1 = np.array([[left_point[0], right_point[0]], [left_point[1], right_point[1]]])

        KeyPoint_for_draw = KeyPoint_3d[:, [0, 2]].copy()
        AverageValue_x = np.mean(KeyPoint[:, 0]) * 100
        AverageValue_y = np.mean(KeyPoint[:, 1]) * 100
        
        # start our pipeline
        # 1. find minimum bbox with special consideration: maximize pc number between current bbox
        #    and its 0.8 times bbox with same bbox center and orientation

        current_angle = 0.0
        min_x = 0
        min_y = 0
        max_x = 100
        max_y = 100

        final = None
        seq = np.arange(0, 90.5 * np.pi / 180, 0.5 * np.pi / 180)
        FinalPoint = np.array([0., 0.])

        if maximum:
            cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX), 1)
        else:
            cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

        while True:
            minValue = -1
            for i in seq:
                try:
                    RotateMatrix = np.array([[np.cos(i), -1 * np.sin(i)],
                                            [np.sin(i), np.cos(i)]])
                    temp = np.dot(KeyPoint, RotateMatrix)
                    current_min_x, current_min_y = np.amin(temp, axis=0)
                    current_max_x, current_max_y = np.amax(temp, axis=0)

                    # construct a sub-rectangle smaller than bounding box, whose x_range and y_range is defined below:
                    thresh_min_x = current_min_x + detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
                    thresh_max_x = current_max_x - detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
                    thresh_min_y = current_min_y + detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)
                    thresh_max_y = current_max_y - detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)

                    thresh_filter_1 = (temp[:, 0] >= thresh_min_x) & (temp[:, 0] <= thresh_max_x)
                    thresh_filter_2 = (temp[:, 1] >= thresh_min_y) & (temp[:, 1] <= thresh_max_y)
                    thresh_filter = (thresh_filter_1 & thresh_filter_2).astype(np.uint8)

                    # calculate satisfying point number between original bbox and shrinked bbox
                    CurrentValue = np.sum(thresh_filter) / temp.shape[0]

                except:
                    print('return none line 425')
                    return None, None, None, None, None, None, None, None

                if CurrentValue < minValue or minValue < 0:
                    final = temp
                    minValue = CurrentValue
                    current_angle = i
                    min_x = current_min_x
                    min_y = current_min_y
                    max_x = current_max_x
                    max_y = current_max_y

            box = np.array([[min_x, min_y],
                            [min_x, max_y],
                            [max_x, max_y],
                            [max_x, min_y]])  # rotate clockwise

            angle = current_angle

            # calculate satisfying bounding box
            box = np.dot(box, np.array([[np.cos(angle), np.sin(angle)],
                                        [-1 * np.sin(angle), np.cos(angle)]])).astype(np.float32)

            index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
            
            # compare which side has the most points, then determine final diagonal, 
            # final key vertex (Current_FinalPoint) and its index (Current_Index) in bbox
            if number_1 < number_2:
                Current_FinalPoint = point_2
                Current_Index = index_2
            else:
                Current_FinalPoint = point_1
                Current_Index = index_1

            # quitting this loop requires:
            # 1. deleting point process has not stopped (cut_times is not positive)
            # 2. after deleting points, key vertex point's location is almost same as that before deleting points   
            if cut_times == 0 and (Current_FinalPoint[0] - FinalPoint[0]) ** 2 + \
                                (Current_FinalPoint[1] - FinalPoint[1]) ** 2 < detect_config.KEY_VERTEX_MOVE_DIST_THRESH:
                break
            else:
                if cut_times == 0:
                    # the end of deleting point process, re-calculate new cut_times with lower number of variable [KeyPoint]
                    FinalPoint = Current_FinalPoint
                    if maximum:
                        cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX_2), 1)
                    else:
                        cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

                else:
                    # continue current deleting point process
                    cut_times -= 1
                    
                    # avoid too fierce deleting
                    if KeyPoint.shape[0] < detect_config.THRESH_MIN_POINTS_AFTER_DELETING:
                        print('return none line 480', KeyPoint.shape[0], detect_config.THRESH_MIN_POINTS_AFTER_DELETING)

                        return None, None, None, None, None, None, None, None
                    
                    index, KeyPoint, final = delete_noisy_point_cloud(final, Current_Index, KeyPoint, 
                                                                    detect_config.DELETE_TIMES_EVERY_EPOCH)

        # while the loop is broken, the variable [box] is the final selected bbox for car point clouds
        index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
        
        # here we get final key-vertex (FinalPoint) and its index in box (FinalIndex)
        if number_1 < number_2:
            FinalPoint = point_2
            FinalIndex = index_2
        else:
            FinalPoint = point_1
            FinalIndex = index_1
            
        # 2. calculate intersection from key-vertex to frustum [vertically]
        # mat_1: rotation matrix from  
        FinalPoint_Weight = np.linalg.inv(mat_1).dot(np.array([FinalPoint[0], FinalPoint[1]]).T)

        top_1 = [box_2d[0], box_2d[1], 1]
        top_2 = [box_2d[2], box_2d[1], 1]
        bot_1 = [box_2d[0], box_2d[3], 1]
        bot_2 = [box_2d[2], box_2d[3], 1]

        # top_1 = np.array([box_2d[0], box_2d[1], 1]).T
        top_1 = torch.tensor(top_1,  dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        top_1 = self.get_geometry_at_image_coords(top_1, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)

        top_2 = torch.tensor(top_2,  dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        top_2 = self.get_geometry_at_image_coords(top_2, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)

        bot_1 = torch.tensor(bot_1,  dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        bot_1 = self.get_geometry_at_image_coords(bot_1, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)

        bot_2 = torch.tensor(bot_2,  dtype=camera2lidar_rots.dtype, device=camera2lidar_rots.device).reshape(1, 3)
        bot_2 = self.get_geometry_at_image_coords(bot_2, [cam_idx], [batch_idx], 
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        ).cpu().numpy().reshape(3)
        # top_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[1], 1]).T)
        # top_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[1], 1]).T)
        # bot_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[3], 1]).T)
        # bot_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[3], 1]).T)
        
        if truncate == False:
            y_min, y_max = Calculate_Height(top_1, top_2, bot_1, bot_2, FinalPoint)
        else:
            # for truncate cases, calculating height from frustum may fail if key-vertex is not inside frustum area
            
            y_min = np.min(KeyPoint_3d[:, 1])
            plane = fitPlane(sample_points)
            eps = 1e-8
            sign = np.sign(np.sign(plane[1]) + 0.5)
            try:
                y_max = -1 * (plane[0] * FinalPoint[0] + plane[2] * FinalPoint[1] - 1) / (plane[1] + eps * sign)
            except:
                y_max = np.max(KeyPoint_3d[:, 1])

        # filter cars with very bad height
        # if np.abs(y_max - y_min) < detect_config.MIN_HEIGHT_NORMAL or \
        # np.abs(y_max - y_min) > detect_config.MAX_HEIGHT_NORMAL or \
        # (truncate == True and (y_max < detect_config.MIN_TOP_TRUNCATE or 
        #                         y_max > detect_config.MAX_TOP_TRUNCATE or 
        #                         y_min < detect_config.MIN_BOT_TRUNCATE or 
        #                         y_min > detect_config.MAX_BOT_TRUNCATE)):
            
        #     error_message = "top: %.4f, bottom: %.4f, car height: %.4f, deprecated" % (y_min, y_max, np.abs(y_max - y_min))
        #     print('return none line 559')

        #     return None, None, None, None, None, None, error_message, 0

        # 3. calculate intersection from key-vertex to frustum [horizontally], to get car's length and width
        if truncate == True or FinalPoint_Weight[0] < detect_config.FINAL_POINT_FLIP_THRESH or \
                            FinalPoint_Weight[1] < detect_config.FINAL_POINT_FLIP_THRESH:
            loc1 = box[FinalIndex - 1]
            loc2 = box[(FinalIndex + 1) % 4]
            loc3 = np.array([0., 0.])
            loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
            loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]
        else:
            loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
                                                                            left_point=left_point,
                                                                            FinalIndex=FinalIndex, FinalPoint=FinalPoint,
                                                                            shape=KeyPoint.shape[0])
            
            weight = np.linalg.inv(mat_1).dot(np.array([loc3[0], loc3[1]]).T)
            
            # correct some cases with failed checking on key-vertex (very close to frustum's left/right side)
            if weight[0] <= detect_config.FINAL_POINT_FLIP_THRESH or weight[1] <= detect_config.FINAL_POINT_FLIP_THRESH:
                if FinalIndex == index_1:
                    FinalIndex = index_2
                    FinalPoint = point_2
                else:
                    FinalIndex = index_1
                    FinalPoint = point_1
                
                # re-calculate intersection
                loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
                                                                                left_point=left_point,
                                                                                FinalIndex=FinalIndex, FinalPoint=FinalPoint,
                                                                                shape=KeyPoint.shape[0])

            # if the angle between bounding box and frustum radiation lines is smaller than detect_config.ANCHOR_FIT_DEGREE_THRESH,
            # ignore the intersection strategy, and use anchor box 
            # (with pre-defined length-width rate, which is medium value in total KITTI dataset)
            
            loc1, loc2, loc3 = check_anchor_fitting(box, loc1, loc2, loc3, angle_1, angle_2, 
                                                    FinalIndex, FinalPoint, y_max, y_min,
                                                    anchor_fit_degree_thresh=detect_config.ANCHOR_FIT_DEGREE_THRESH, 
                                                    height_width_rate=detect_config.HEIGHT_WIDTH_RATE, 
                                                    height_length_rate=detect_config.HEIGHT_LENGTH_RATE,
                                                    length_width_boundary=detect_config.LENGTH_WIDTH_BOUNDARY)
            
        # 4. filter cases with still bad key-vertex definition,
        # we assume that key-vertex must be in the top 2 nearest to camera along z axis
        
        z_less_than_finalpoint = 0
        for i in range(len(box)):
            if i == FinalIndex:
                continue
            if box[i, 1] < box[FinalIndex, 1]:
                z_less_than_finalpoint += 1
        
        if z_less_than_finalpoint >= 2:
            error_message = "keypoint error, deprecated."
            print('return none line 617')

            return None, None, None, None, None, None, error_message, 2

        len_1 = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
        len_2 = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)

        car_length = max(len_1, len_2)
        car_width  = min(len_1, len_2)

        # define max(len_1, len_2) as length of the car, and min(len_1, len_2) as width of the car
        # length of the car is 3.0-5.0m, and width of the car is 1.2-2.2m
        # filter cars with very bad length or height
        
        # if not (detect_config.MIN_WIDTH  <= car_width  <= detect_config.MAX_WIDTH) or \
        # not (detect_config.MIN_LENGTH <= car_length <= detect_config.MAX_LENGTH):
        #     print('return none 635 (not within bounds)')
        #     error_message = "length: %.4f, width: %.4f, deprecated" % (car_length, car_width)
        #     return None, None, None, None, None, None, error_message, 1

        KeyPoint_side = KeyPoint_3d[:, [0, 1]].copy()
        img_side = np.zeros((700, 700, 3), 'f4')
        
        # draw gt bounding box from 3D to 2D plane
        # img = draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y)
        fig, ax = plt.subplots()

        # draw frustum's left and right line in 2D plane
        img = draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y)

        # ax.plot([AverageValue_x, left_point[0]], [AverageValue_y, left_point[1]], label='left frust')
        # ax.plot([AverageValue_x, right_point[0]], [AverageValue_y, right_point[1]], label='right frust')
        
        # draw psuedo bounding box from 3D to 2D plane [before] calculating intersection
        img = draw_bbox_3d_to_2d_psuedo_no_intersection(img, box, AverageValue_x, AverageValue_y)

        xs = [box[i][0] for i in [0, 1, 2, 3, 0]]
        ys = [box[i][1] for i in [0, 1, 2, 3, 0]]

        ax.plot(xs, ys, label='box draw')

        # draw psuedo bounding box from 3D to 2D plane [after] calculating intersection
        img = draw_bbox_3d_to_2d_psuedo_with_key_vertex(img, FinalPoint, loc1, loc2, loc3, AverageValue_x, AverageValue_y)

        ax.scatter(loc1[0], loc1[1], label='loc1')
        ax.scatter(loc2[0], loc2[1], label='loc2')
        ax.scatter(loc3[0], loc3[1], label='loc3')
        ax.scatter(FinalPoint[0], FinalPoint[1], label='FinalPoint')
        
        # draw car point clouds after region growth
        img = draw_point_clouds(img, KeyPoint_for_draw, AverageValue_x, AverageValue_y)

        ax.scatter([x[0] for x in KeyPoint_for_draw], [x[1] for x in KeyPoint_for_draw], label='pc')
        x1 = min([x[0] for x in KeyPoint_for_draw])
        x2 = max([x[0] for x in KeyPoint_for_draw])
        y1 = min([x[1] for x in KeyPoint_for_draw])
        y2 = max([x[1] for x in KeyPoint_for_draw])

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        plt.legend()
        plt.savefig('fgr_down_ax.png', bbox_inches='tight')
        
        return img, img_side, FinalPoint, loc1, loc3, loc2, y_max, y_min

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

    def project_to_camera(self, batch_dict, points, batch_idx=0, cam_idx=0):
        # do projection to multi-view images and return a mask of which images the points lay on
        cur_coords = points.clone()

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
        bboxes = self.get_bboxes(batch_dict)
        batch_dict['final_box_dicts'] = bboxes

        assert not self.training, "not trainable!"
        return batch_dict

    def get_bboxes(self, batch_dict):
        proposed_boxes, proposed_labels, proposed_scores, proposed_batch_idx = self.get_proposals(batch_dict)

        empty_dict = dict(pred_boxes=[], pred_scores=[], pred_labels=[])
        ret_dict = [empty_dict] * batch_dict['batch_size']


        for k in range(batch_dict['batch_size']):
            mask = (proposed_batch_idx == k)

            ret_dict[k]['pred_boxes'] = proposed_boxes[mask]#.cpu()
            ret_dict[k]['pred_scores'] = proposed_scores[mask]
            ret_dict[k]['pred_labels'] = proposed_labels[mask].int() # + 1 is done in preprocessed_detector

        return ret_dict 

