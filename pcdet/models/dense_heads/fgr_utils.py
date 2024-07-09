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

from ..model_utils import model_nms_utils
import time
import cv2

import traceback


def draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y):
    
    gt_box_3d = corner.copy()
    for point in gt_box_3d:
        point[0] = point[0] * 100 + 250 - AverageValue_x
        point[2] = point[2] * 100 + 250 - AverageValue_y

    cv2.line(img, (int(gt_box_3d[0][0]), int(gt_box_3d[0][2])), (int(gt_box_3d[1][0]), int(gt_box_3d[1][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[1][0]), int(gt_box_3d[1][2])), (int(gt_box_3d[2][0]), int(gt_box_3d[2][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[2][0]), int(gt_box_3d[2][2])), (int(gt_box_3d[3][0]), int(gt_box_3d[3][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[3][0]), int(gt_box_3d[3][2])), (int(gt_box_3d[0][0]), int(gt_box_3d[0][2])), (0, 255, 255), 1, 4)
    
    return img


def draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y):
    print('draw_frustum_lr_line', left_point, right_point, AverageValue_x, AverageValue_y)

    left_point_draw = np.array([0., 0.])
    left_point_draw[0] = (left_point[0] * 20000 + 250 - AverageValue_x)
    left_point_draw[1] = (left_point[1] * 20000 + 250 - AverageValue_y)

    right_point_draw = np.array([0., 0.])
    right_point_draw[0] = (right_point[0] * 20000 + 250 - AverageValue_x)
    right_point_draw[1] = (right_point[1] * 20000 + 250 - AverageValue_y)

    initial_point_draw = np.array([0., 0.])
    initial_point_draw[0] = 250 - AverageValue_x
    initial_point_draw[1] = 250 - AverageValue_y

    print('initial_point_draw', initial_point_draw)
    print('left_point_draw', left_point_draw)

    cv2.line(img, tuple(initial_point_draw.astype('int')), tuple(left_point_draw.astype('int')),
             (255, 255, 0), 1, 4)
    cv2.line(img, tuple(initial_point_draw.astype('int')), tuple(right_point_draw.astype('int')),
             (255, 255, 0), 1, 4)

    return img


def draw_bbox_3d_to_2d_psuedo_no_intersection(img, box_no_intersection, AverageValue_x, AverageValue_y):
    
    box_draw = box_no_intersection.copy()
    
    for var in box_draw:
        var[0] = var[0] * 100 + 250 - AverageValue_x
        var[1] = var[1] * 100 + 250 - AverageValue_y
    
    # print('box draw', box_draw)
    box_draw = [[int(y) for y in x] for x in box_draw]
    cv2.line(img, tuple(box_draw[0]), tuple(box_draw[1]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[1]), tuple(box_draw[2]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[2]), tuple(box_draw[3]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[3]), tuple(box_draw[0]), (255, 0, 255), 1, 4)
    
    return img


def draw_bbox_3d_to_2d_psuedo_with_key_vertex(img, FinalPoint, loc1, loc2, loc3, AverageValue_x, AverageValue_y):
    
    loc1_draw = np.array([0., 0.])
    loc2_draw = np.array([0., 0.])
    loc3_draw = np.array([0., 0.])
    loc1_draw[0] = loc1[0] * 100 + 250 - AverageValue_x
    loc1_draw[1] = loc1[1] * 100 + 250 - AverageValue_y
    loc2_draw[0] = loc2[0] * 100 + 250 - AverageValue_x
    loc2_draw[1] = loc2[1] * 100 + 250 - AverageValue_y
    loc3_draw[0] = loc3[0] * 100 + 250 - AverageValue_x
    loc3_draw[1] = loc3[1] * 100 + 250 - AverageValue_y
    
    # draw key vertex with larger point than normal point cloud
    FinalPoint_draw = np.array([0., 0.])
    FinalPoint_draw[0] = FinalPoint[0] * 100 + 250 - AverageValue_x
    FinalPoint_draw[1] = FinalPoint[1] * 100 + 250 - AverageValue_y
    cv2.circle(img, tuple(FinalPoint_draw.astype('int')), 3, (0, 255, 0), 4)

    cv2.line(img, tuple(loc1_draw.astype('int')), tuple(FinalPoint_draw.astype('int')), (0, 0, 0), 1, 4)
    cv2.line(img, tuple(loc1_draw.astype('int')), tuple(loc3_draw.astype('int')), (0, 0, 255), 1, 4)
    cv2.line(img, tuple(loc3_draw.astype('int')), tuple(loc2_draw.astype('int')), (0, 0, 255), 1, 4)
    cv2.line(img, tuple(loc2_draw.astype('int')), tuple(FinalPoint_draw.astype('int')), (0, 0, 255), 1, 4)
    
    return img

def draw_point_clouds(img, KeyPoint_for_draw, AverageValue_x, AverageValue_y):

    for point in KeyPoint_for_draw:
        a = point[0] * 100 + 250 - AverageValue_x
        b = point[1] * 100 + 250 - AverageValue_y
        cv2.circle(img, (int(a), int(b)), 1, (255, 255, 255), 2)
    
    return img


def draw_2d_box_in_2d_image(KeyPoint_3d, box_2d, p2):

    img = np.zeros((1000, 1000, 3), 'f4')

    KeyPoint_for_draw = np.copy(KeyPoint_3d[:, [0, 2]])
    KeyPoint = KeyPoint_3d[:, [0, 2]]

    AverageValue_x = np.mean(KeyPoint[:, 0]) * 100
    AverageValue_y = np.mean(KeyPoint[:, 1]) * 100

    for point in KeyPoint_for_draw:
        a = point[0] * 50 + 500 - AverageValue_x
        b = point[1] * 50 + 100 - AverageValue_y
        cv2.circle(img, (int(a), int(b)), 1, (255, 255, 255), 0)

    left_point = np.array([box_2d[0], 0, 1])
    right_point = np.array([box_2d[2], 0, 1])

    left_point = np.linalg.inv(p2[:, [0, 1, 2]]).dot(left_point.T)
    right_point = np.linalg.inv(p2[:, [0, 1, 2]]).dot(right_point.T)

    left_point = left_point[[0, 2]]
    right_point = right_point[[0, 2]]

    left_point_draw = np.array([0., 0.])
    right_point_draw = np.array([0., 0.])

    while True:
        zoom_factor = 60000
        try:
            left_point_draw[0] = (left_point[0] * zoom_factor + 100 - AverageValue_x)
            left_point_draw[1] = (left_point[1] * zoom_factor + 100 - AverageValue_y)

            right_point_draw[0] = (right_point[0] * zoom_factor + 100 - AverageValue_x)
            right_point_draw[1] = (right_point[1] * zoom_factor + 100 - AverageValue_y)
        except OverflowError:
            zoom_factor /= 6
            continue
        else:
            break

    initial_point_draw = np.array([0., 0.])
    initial_point_draw[0] = 500 - AverageValue_x
    initial_point_draw[1] = 100 - AverageValue_y

    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(left_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)
    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(right_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)

    return img

def draw_3d_box_in_2d_image(img, box):

    # draw 3D box in 2D RGB image, if needed

    cv2.line(img, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[1][0]), int(box[1][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[2][0]), int(box[2][1])), (int(box[3][0]), int(box[3][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[3][0]), int(box[3][1])), (int(box[0][0]), int(box[0][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[0][0]), int(box[0][1])), (int(box[4][0]), int(box[4][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[1][0]), int(box[1][1])), (int(box[5][0]), int(box[5][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[2][0]), int(box[2][1])), (int(box[6][0]), int(box[6][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[3][0]), int(box[3][1])), (int(box[7][0]), int(box[7][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[4][0]), int(box[4][1])), (int(box[5][0]), int(box[5][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[5][0]), int(box[5][1])), (int(box[6][0]), int(box[6][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[6][0]), int(box[6][1])), (int(box[7][0]), int(box[7][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[7][0]), int(box[7][1])), (int(box[4][0]), int(box[4][1])), (0, 0, 255), 1, 4)

    return img

def show_velodyne_in_camera(loc0, loc1, loc2, loc3, y_min, y_max):

    # use mayavi to draw 3D bbox 
    corners = np.array([[[loc0[0], loc1[0], loc2[0], loc3[0], loc0[0], loc1[0], loc2[0], loc3[0]],
                         [loc0[1], loc1[1], loc2[1], loc3[1], loc0[1], loc1[1], loc2[1], loc3[1]],
                         [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max]]],
                       dtype=np.float32)

    for i in range(corners.shape[0]):
        corner = corners[i]
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        mayavi.mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)

""" From FGR/FGR/detect.py"""
# def Find_2d_box(maximum, KeyPoint_3d, box_2d, p2, corner, detect_config, 
#                 total_pc=None, cluster=None, truncate=False, sample_points=None):
#     # corner is GT corners and not necessary

#     # ignore less point cases
#     if len(KeyPoint_3d) < 10:
#         return None, None, None, None, None, None, None, None
    
#     img = np.zeros((700, 700, 3), 'f4')
#     KeyPoint = KeyPoint_3d[:, [0, 2]].copy()

#     left_point  = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[0], 0, 1]).copy().T)[[0, 2]]
#     right_point = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[2], 0, 1]).copy().T)[[0, 2]]

#     mat_1 = np.array([[left_point[0], right_point[0]], [left_point[1], right_point[1]]])

#     KeyPoint_for_draw = KeyPoint_3d[:, [0, 2]].copy()
#     AverageValue_x = np.mean(KeyPoint[:, 0]) * 100
#     AverageValue_y = np.mean(KeyPoint[:, 1]) * 100
    
#     # start our pipeline
#     # 1. find minimum bbox with special consideration: maximize pc number between current bbox
#     #    and its 0.8 times bbox with same bbox center and orientation

#     current_angle = 0.0
#     min_x = 0
#     min_y = 0
#     max_x = 100
#     max_y = 100

#     final = None
#     seq = np.arange(0, 90.5 * np.pi / 180, 0.5 * np.pi / 180)
#     FinalPoint = np.array([0., 0.])

#     if maximum:
#         cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX), 1)
#     else:
#         cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

#     while True:
#         minValue = -1
#         for i in seq:
#             try:
#                 RotateMatrix = np.array([[np.cos(i), -1 * np.sin(i)],
#                                          [np.sin(i), np.cos(i)]])
#                 temp = np.dot(KeyPoint, RotateMatrix)
#                 current_min_x, current_min_y = np.amin(temp, axis=0)
#                 current_max_x, current_max_y = np.amax(temp, axis=0)

#                 # construct a sub-rectangle smaller than bounding box, whose x_range and y_range is defined below:
#                 thresh_min_x = current_min_x + detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
#                 thresh_max_x = current_max_x - detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
#                 thresh_min_y = current_min_y + detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)
#                 thresh_max_y = current_max_y - detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)

#                 thresh_filter_1 = (temp[:, 0] >= thresh_min_x) & (temp[:, 0] <= thresh_max_x)
#                 thresh_filter_2 = (temp[:, 1] >= thresh_min_y) & (temp[:, 1] <= thresh_max_y)
#                 thresh_filter = (thresh_filter_1 & thresh_filter_2).astype(np.uint8)

#                 # calculate satisfying point number between original bbox and shrinked bbox
#                 CurrentValue = np.sum(thresh_filter) / temp.shape[0]

#             except:
#                 return None, None, None, None, None, None, None, None

#             if CurrentValue < minValue or minValue < 0:
#                 final = temp
#                 minValue = CurrentValue
#                 current_angle = i
#                 min_x = current_min_x
#                 min_y = current_min_y
#                 max_x = current_max_x
#                 max_y = current_max_y

#         box = np.array([[min_x, min_y],
#                         [min_x, max_y],
#                         [max_x, max_y],
#                         [max_x, min_y]])  # rotate clockwise

#         angle = current_angle

#         # calculate satisfying bounding box
#         box = np.dot(box, np.array([[np.cos(angle), np.sin(angle)],
#                                     [-1 * np.sin(angle), np.cos(angle)]])).astype(np.float32)

#         index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
        
#         # compare which side has the most points, then determine final diagonal, 
#         # final key vertex (Current_FinalPoint) and its index (Current_Index) in bbox
#         if number_1 < number_2:
#             Current_FinalPoint = point_2
#             Current_Index = index_2
#         else:
#             Current_FinalPoint = point_1
#             Current_Index = index_1

#         # quitting this loop requires:
#         # 1. deleting point process has not stopped (cut_times is not positive)
#         # 2. after deleting points, key vertex point's location is almost same as that before deleting points   
#         if cut_times == 0 and (Current_FinalPoint[0] - FinalPoint[0]) ** 2 + \
#                               (Current_FinalPoint[1] - FinalPoint[1]) ** 2 < detect_config.KEY_VERTEX_MOVE_DIST_THRESH:
#             break
#         else:
#             if cut_times == 0:
#                 # the end of deleting point process, re-calculate new cut_times with lower number of variable [KeyPoint]
#                 FinalPoint = Current_FinalPoint
#                 if maximum:
#                     cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX_2), 1)
#                 else:
#                     cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

#             else:
#                 # continue current deleting point process
#                 cut_times -= 1
                
#                 # avoid too fierce deleting
#                 if KeyPoint.shape[0] < detect_config.THRESH_MIN_POINTS_AFTER_DELETING:
#                     return None, None, None, None, None, None, None, None
                
#                 index, KeyPoint, final = delete_noisy_point_cloud(final, Current_Index, KeyPoint, 
#                                                                   detect_config.DELETE_TIMES_EVERY_EPOCH)

#     # while the loop is broken, the variable [box] is the final selected bbox for car point clouds
#     index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
    
#     # here we get final key-vertex (FinalPoint) and its index in box (FinalIndex)
#     if number_1 < number_2:
#         FinalPoint = point_2
#         FinalIndex = index_2
#     else:
#         FinalPoint = point_1
#         FinalIndex = index_1
        
#     # 2. calculate intersection from key-vertex to frustum [vertically]
#     # mat_1: rotation matrix from  
#     FinalPoint_Weight = np.linalg.inv(mat_1).dot(np.array([FinalPoint[0], FinalPoint[1]]).T)

#     top_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[1], 1]).T)
#     top_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[1], 1]).T)
#     bot_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[3], 1]).T)
#     bot_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[3], 1]).T)
    
#     if truncate == False:
#         y_min, y_max = Calculate_Height(top_1, top_2, bot_1, bot_2, FinalPoint)
#     else:
#         # for truncate cases, calculating height from frustum may fail if key-vertex is not inside frustum area
        
#         y_min = np.min(KeyPoint_3d[:, 1])
#         plane = fitPlane(sample_points)
#         eps = 1e-8
#         sign = np.sign(np.sign(plane[1]) + 0.5)
#         try:
#             y_max = -1 * (plane[0] * FinalPoint[0] + plane[2] * FinalPoint[1] - 1) / (plane[1] + eps * sign)
#         except:
#             y_max = np.max(KeyPoint_3d[:, 1])

#     # filter cars with very bad height
#     if np.abs(y_max - y_min) < detect_config.MIN_HEIGHT_NORMAL or \
#        np.abs(y_max - y_min) > detect_config.MAX_HEIGHT_NORMAL or \
#        (truncate == True and (y_max < detect_config.MIN_TOP_TRUNCATE or 
#                               y_max > detect_config.MAX_TOP_TRUNCATE or 
#                               y_min < detect_config.MIN_BOT_TRUNCATE or 
#                               y_min > detect_config.MAX_BOT_TRUNCATE)):
        
#         error_message = "top: %.4f, bottom: %.4f, car height: %.4f, deprecated" % (y_min, y_max, np.abs(y_max - y_min))
#         return None, None, None, None, None, None, error_message, 0

#     # 3. calculate intersection from key-vertex to frustum [horizontally], to get car's length and width
#     if truncate == True or FinalPoint_Weight[0] < detect_config.FINAL_POINT_FLIP_THRESH or \
#                            FinalPoint_Weight[1] < detect_config.FINAL_POINT_FLIP_THRESH:
#         loc1 = box[FinalIndex - 1]
#         loc2 = box[(FinalIndex + 1) % 4]
#         loc3 = np.array([0., 0.])
#         loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
#         loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]
#     else:
#         loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
#                                                                         left_point=left_point,
#                                                                         FinalIndex=FinalIndex, FinalPoint=FinalPoint,
#                                                                         shape=KeyPoint.shape[0])
        
#         weight = np.linalg.inv(mat_1).dot(np.array([loc3[0], loc3[1]]).T)
        
#         # correct some cases with failed checking on key-vertex (very close to frustum's left/right side)
#         if weight[0] <= detect_config.FINAL_POINT_FLIP_THRESH or weight[1] <= detect_config.FINAL_POINT_FLIP_THRESH:
#             if FinalIndex == index_1:
#                 FinalIndex = index_2
#                 FinalPoint = point_2
#             else:
#                 FinalIndex = index_1
#                 FinalPoint = point_1
            
#             # re-calculate intersection
#             loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
#                                                                             left_point=left_point,
#                                                                             FinalIndex=FinalIndex, FinalPoint=FinalPoint,
#                                                                             shape=KeyPoint.shape[0])

#         # if the angle between bounding box and frustum radiation lines is smaller than detect_config.ANCHOR_FIT_DEGREE_THRESH,
#         # ignore the intersection strategy, and use anchor box 
#         # (with pre-defined length-width rate, which is medium value in total KITTI dataset)
        
#         loc1, loc2, loc3 = check_anchor_fitting(box, loc1, loc2, loc3, angle_1, angle_2, 
#                                                 FinalIndex, FinalPoint, y_max, y_min,
#                                                 anchor_fit_degree_thresh=detect_config.ANCHOR_FIT_DEGREE_THRESH, 
#                                                 height_width_rate=detect_config.HEIGHT_WIDTH_RATE, 
#                                                 height_length_rate=detect_config.HEIGHT_LENGTH_RATE,
#                                                 length_width_boundary=detect_config.LENGTH_WIDTH_BOUNDARY)
        
#     # 4. filter cases with still bad key-vertex definition,
#     # we assume that key-vertex must be in the top 2 nearest to camera along z axis
    
#     z_less_than_finalpoint = 0
#     for i in range(len(box)):
#         if i == FinalIndex:
#             continue
#         if box[i, 1] < box[FinalIndex, 1]:
#             z_less_than_finalpoint += 1
    
#     if z_less_than_finalpoint >= 2:
#         error_message = "keypoint error, deprecated."
#         return None, None, None, None, None, None, error_message, 2

#     len_1 = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
#     len_2 = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)

#     car_length = max(len_1, len_2)
#     car_width  = min(len_1, len_2)

#     # define max(len_1, len_2) as length of the car, and min(len_1, len_2) as width of the car
#     # length of the car is 3.0-5.0m, and width of the car is 1.2-2.2m
#     # filter cars with very bad length or height
    
#     if not (detect_config.MIN_WIDTH  <= car_width  <= detect_config.MAX_WIDTH) or \
#        not (detect_config.MIN_LENGTH <= car_length <= detect_config.MAX_LENGTH):
#         error_message = "length: %.4f, width: %.4f, deprecated" % (car_length, car_width)
#         return None, None, None, None, None, None, error_message, 1

#     KeyPoint_side = KeyPoint_3d[:, [0, 1]].copy()
#     img_side = np.zeros((700, 700, 3), 'f4')
    
#     # draw gt bounding box from 3D to 2D plane
#     # img = draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y)

#     # draw frustum's left and right line in 2D plane
#     img = draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y)
    
#     # draw psuedo bounding box from 3D to 2D plane [before] calculating intersection
#     img = draw_bbox_3d_to_2d_psuedo_no_intersection(img, box, AverageValue_x, AverageValue_y)

#     # draw psuedo bounding box from 3D to 2D plane [after] calculating intersection
#     img = draw_bbox_3d_to_2d_psuedo_with_key_vertex(img, FinalPoint, loc1, loc2, loc3, AverageValue_x, AverageValue_y)
    
#     # draw car point clouds after region growth
#     img = draw_point_clouds(img, KeyPoint_for_draw, AverageValue_x, AverageValue_y)
    
#     return img, img_side, FinalPoint, loc1, loc3, loc2, y_max, y_min


def delete_noisy_point_cloud(final, Current_Index, KeyPoint, delete_times_every_epoch=2):
                
    # re-calculate key-vertex's location
    # KeyPoint: original point cloud
    # final: [rotated] point cloud
    # deleting method: from KeyPoint, calculate the point with maximum/minimum x and y,
    # extract their indexes, and delete them from numpy.array
    # one basic assumption on box's location order is: 0 to 3 => left-bottom to left_top (counter-clockwise)
    if Current_Index == 2 or Current_Index == 3:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 0], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 0 or Current_Index == 1:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 0], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 1 or Current_Index == 2:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 1], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 0 or Current_Index == 3:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 1], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)
    
    return index, KeyPoint, final


def find_key_vertex_by_pc_number(KeyPoint, box):
    # first diagonal: (box[1], box[3]), corresponding points: box[0] / box[2]
    # (... > 0) is the constraint for key vertex's side towards the diagnoal
    if box[0][0] * (box[1][1] - box[3][1]) - box[0][1] * (box[1][0] - box[3][0]) + (
            box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0:
        index_1 = 0
    else:
        index_1 = 2

    # first diagonal: (box[1], box[3]), and calculate the point number on one side of this diagonal,
    # (... > 0) to constraint the current side, which is equal to the side of key vertex (box[index_1]) 
    filter_1 = (KeyPoint[:, 0] * (box[1][1] - box[3][1]) - KeyPoint[:, 1] * (box[1][0] - box[3][0]) + (
                box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0)
    number_1 = np.sum(filter_1)
        
    # find which side contains more points, record this side and corresponding point number, 
    # and key vertex, towards current diagonal (box[1], box[3])
        
    # number_1: most point number
    # index_1:  corresponding key vertex's index of bbox points
    # point_1:  corresponding key vertex
        
    if number_1 < KeyPoint.shape[0] / 2:
        number_1 = KeyPoint.shape[0] - number_1
        index_1 = (index_1 + 2) % 4

    point_1 = box[index_1]

    # second diagonal: (box[0], box[2]), corresponding points: box[1] / box[3]
    # (... > 0) to constraint the current side, which is equal to the side of key vertex (box[index_2]) 
    if box[1][0] * (box[0][1] - box[2][1]) - box[2][1] * (box[0][0] - box[2][0]) + (
            box[0][0] * box[2][1] - box[0][1] * box[2][0]) > 0:
        index_2 = 1
    else:
        index_2 = 3

    # find which side contains more points, record this side and corresponding point number, 
    # and key vertex, towards current diagonal (box[0], box[2])
        
    # number_2: most point number
    # index_2:  corresponding key vertex's index of bbox points
    # point_2:  corresponding key vertex
        
    filter_2 = (KeyPoint[:, 0] * (box[0][1] - box[2][1]) - KeyPoint[:, 1] * (box[0][0] - box[2][0]) + (
                box[0][0] * box[2][1] - box[0][1] * box[2][0]) > 0)
    number_2 = np.sum(filter_2)

    if number_2 < KeyPoint.shape[0] / 2:
        number_2 = KeyPoint.shape[0] - number_2
        index_2 = (index_2 + 2) % 4

    point_2 = box[index_2]
    
    return index_1, index_2, point_1, point_2, number_1, number_2


def check_anchor_fitting(box, loc1, loc2, loc3, angle_1, angle_2, FinalIndex, FinalPoint, y_max, y_min,
                         anchor_fit_degree_thresh=10,
                         height_width_rate=0.9305644265920366,
                         height_length_rate=0.3969212090597959, 
                         length_width_boundary=2.2):
        
    if loc1[0] == box[FinalIndex - 1][0] or angle_1 * 180 / np.pi < anchor_fit_degree_thresh or (
            loc1[0] - FinalPoint[0] > 0 and box[FinalIndex - 1][0] - FinalPoint[0] < 0) or \
            (loc1[0] - FinalPoint[0] < 0 and box[FinalIndex - 1][0] - FinalPoint[0] > 0):
        current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            
        # if current_distance is larger than 2.2, we assume current boundary is length, otherwise width,
        # then use length-width rate to calculate another boundary
        if current_distance > length_width_boundary:
            current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
            # ... np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2) * width_length_rate ....
            loc1[0] = FinalPoint[0] + (loc1[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
            loc1[1] = FinalPoint[1] + (loc1[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
        else:
            current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
            loc1[0] = FinalPoint[0] + (loc1[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance
            loc1[1] = FinalPoint[1] + (loc1[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance

        loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
        loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]

    # check another boundary radiated from key vertex
    elif loc2[0] == box[(FinalIndex + 1) % 4][0] or angle_2 * 180 / np.pi < anchor_fit_degree_thresh or (
            loc2[0] - FinalPoint[0] > 0 and box[(FinalIndex + 1) % 4][0] - FinalPoint[0] < 0) or \
            (loc2[0] - FinalPoint[0] < 0 and box[(FinalIndex + 1) % 4][0] - FinalPoint[0] > 0):
        current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
        if current_distance > length_width_boundary:
            current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            loc2[0] = FinalPoint[0] + (loc2[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
            loc2[1] = FinalPoint[1] + (loc2[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
        else:
            current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            loc2[0] = FinalPoint[0] + (loc2[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance
            loc2[1] = FinalPoint[1] + (loc2[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance

        loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
        loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]
    
    return loc1, loc2, loc3

def iou_3d(box_3d, loc0, loc1, loc2, loc3, y_min, y_max):

    # use official code: iou_3d_python.py to calculate 3d iou
    std_box_3d = np.array([[loc1[0], y_max, loc1[1]],
                           [loc0[0], y_max, loc0[1]],
                           [loc3[0], y_max, loc3[1]],
                           [loc2[0], y_max, loc2[1]],
                           [loc1[0], y_min, loc1[1]],
                           [loc0[0], y_min, loc0[1]],
                           [loc3[0], y_min, loc3[1]],
                           [loc2[0], y_min, loc2[1]]])
    std_iou, iou_2d = iou_3d_python.box3d_iou(box_3d, std_box_3d)
    return None, std_iou

def Calculate_Height(top_1, top_2, bot_1, bot_2, keypoint):

    # calculate the [vertical] height in frustum at key vertex (input variable [keypoint])

    # because top and bottom plane of frustum crosses (0, 0, 0), we assume the plane equation: Ax + By + 1 * z = 0
    # |x1 y1| |A|     |-1|         |A|     |x1 y1| -1    |-1|
    # |     | | |  =  |  |     =>  | |  =  |     |    *  |  |
    # |x2 y2| |B|     |-1|         |B|     |x2 y2|       |-1|

    mat_1 = np.array([[top_1[0], top_1[1]], [top_2[0], top_2[1]]])
    mat_2 = np.array([[bot_1[0], bot_1[1]], [bot_2[0], bot_2[1]]])
    mat_3 = np.array([-1., -1.]).T

    top_plane_info = np.linalg.inv(mat_1).dot(mat_3)
    bot_plane_info = np.linalg.inv(mat_2).dot(mat_3)

    top_y = -1 * (keypoint[0] * top_plane_info[0] + keypoint[1] * 1) / top_plane_info[1]
    bot_y = -1 * (keypoint[0] * bot_plane_info[0] + keypoint[1] * 1) / bot_plane_info[1]

    return top_y, bot_y

def Find_Intersection_Point(box, FinalIndex, right_point, left_point, FinalPoint, shape):

    # calculate the [expanded] bounding box from input variable [box], 
    # with intersection line radiated from key vertex (FinalPoint)
    # calculate two line's intersection point by line function:
    # y1 = k * x1 + b
    # y2 = k * x2 + b, 
    # solve these equations

    equation_1_left = np.array(
        [[box[FinalIndex - 1][1] - box[FinalIndex][1], box[FinalIndex][0] - box[FinalIndex - 1][0]],
         [left_point[1], -1 * left_point[0]]])
    equation_1_right = np.array(
        [box[FinalIndex][0] * box[FinalIndex - 1][1] - box[FinalIndex - 1][0] * box[FinalIndex][1], 0])

    try:
        loc1 = np.linalg.inv(equation_1_left).dot(equation_1_right.T)
    except:
        # if there are two parallel lines, np.linalg.inv will fail, so deprecate this case
        return None, None, None

    # determine how to intersect
    # the line radiated from key vertex may cross left frustum and right frustum at the same time, causing two intersection points,
    # so just check which intersection point is right
    # solve this matter still by line equations
    if (loc1[0] - FinalPoint[0]) * (box[FinalIndex - 1][0] - FinalPoint[0]) + \
            (loc1[1] - FinalPoint[1]) * (box[FinalIndex - 1][1] - FinalPoint[1]) > 0:
        vector_1 = np.array([loc1[0] - FinalPoint[0], loc1[1] - FinalPoint[1]])
        angle_1 = np.abs(np.arcsin((left_point[0] * vector_1[1] - left_point[1] * vector_1[0]) / 
                                   (np.linalg.norm(vector_1) * np.linalg.norm(left_point))))
    else:
        equation_2_left = np.array(
            [[box[FinalIndex - 1][1] - box[FinalIndex][1], box[FinalIndex][0] - box[FinalIndex - 1][0]],
             [right_point[1], -1 * right_point[0]]])
        equation_2_right = np.array(
            [box[FinalIndex][0] * box[FinalIndex - 1][1] - box[FinalIndex - 1][0] * box[FinalIndex][1], 0])

        loc1 = np.linalg.inv(equation_2_left).dot(equation_2_right.T)
        vector_1 = np.array([loc1[0] - FinalPoint[0], loc1[1] - FinalPoint[1]])
        angle_1 = np.abs(np.arcsin((right_point[0] * vector_1[1] - right_point[1] * vector_1[0]) / (
                    np.linalg.norm(vector_1) * np.linalg.norm(right_point))))

        if (loc1[0] - FinalPoint[0]) * (box[FinalIndex - 1][0] - FinalPoint[0]) + \
           (loc1[1] - FinalPoint[1]) * (box[FinalIndex - 1][1] - FinalPoint[1]) < 0:
            loc1 = box[FinalIndex - 1].copy()

    equation_1_left = np.array(
        [[box[(FinalIndex + 1) % 4][1] - box[FinalIndex][1], box[FinalIndex][0] - box[(FinalIndex + 1) % 4][0]],
         [right_point[1], -1 * right_point[0]]])
    equation_1_right = np.array(
        [box[FinalIndex][0] * box[(FinalIndex + 1) % 4][1] - box[(FinalIndex + 1) % 4][0] * box[FinalIndex][1], 0])

    loc2 = np.linalg.inv(equation_1_left).dot(equation_1_right.T)

    if (loc2[0] - FinalPoint[0]) * (box[(FinalIndex + 1) % 4][0] - FinalPoint[0]) + \
       (loc2[1] - FinalPoint[1]) * (box[(FinalIndex + 1) % 4][1] - FinalPoint[1]) > 0:
        vector_2 = np.array([loc2[0] - FinalPoint[0], loc2[1] - FinalPoint[1]])
        angle_2 = np.abs(np.arcsin((right_point[0] * vector_2[1] - right_point[1] * vector_2[0]) / (
                np.linalg.norm(vector_2) * np.linalg.norm(right_point))))
    else:
        equation_2_left = np.array(
            [[box[(FinalIndex + 1) % 4][1] - box[FinalIndex][1], box[FinalIndex][0] - box[(FinalIndex + 1) % 4][0]],
             [left_point[1], -1 * left_point[0]]])
        equation_2_right = np.array(
            [box[FinalIndex][0] * box[(FinalIndex + 1) % 4][1] - box[(FinalIndex + 1) % 4][0] * box[FinalIndex][1], 0])

        loc2 = np.linalg.inv(equation_2_left).dot(equation_2_right.T)
        vector_2 = np.array([loc2[0] - FinalPoint[0], loc2[1] - FinalPoint[1]])
        angle_2 = np.abs(np.arcsin((left_point[0] * vector_2[1] - left_point[1] * vector_2[0]) / (
                np.linalg.norm(vector_2) * np.linalg.norm(left_point))))

        if (loc2[0] - FinalPoint[0]) * (box[(FinalIndex + 1) % 4][0] - FinalPoint[0]) + \
                (loc2[1] - FinalPoint[1]) * (box[(FinalIndex + 1) % 4][1] - FinalPoint[1]) < 0:
            loc2 = box[(FinalIndex + 1) % 4].copy()


    # infer the last point location (loc3) from other 3 points (loc1, loc2, FinalPoint)
    loc3 = np.array([0., 0.])
    loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
    loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]

    return loc1, loc2, loc3, angle_1, angle_2
""" above modified from FGR/FGR/detect.py"""

# modified from kitti_utils_official
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]

def check_parallel(points):
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    p = (a + b + c) / 2
    
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    if area < 1e-2:
        return True
    else:
        return False

def calculate_ground(point_cloud, thresh_ransac=0.15, back_cut=True, back_cut_z=-5.0):
    # Only keep points in front of camera (positive z)
    if back_cut:
        point_cloud = point_cloud[point_cloud[:,2] > back_cut_z]   # camera frame 3 x N

    planeDiffThreshold = thresh_ransac
    # temp = np.sort(point_cloud[:,1])[int(point_cloud.shape[0]*0.75)]
    # temp = np.sort(point_cloud[:,1])[int(point_cloud.shape[0]*0.25)]
    temp = 0.0
    print('temp', temp)
    print('point_cloud', point_cloud.shape)
    cloud = point_cloud[point_cloud[:,1]<temp]
    print('cloud', cloud.shape)
    for i in range(3):
        print('coud_',i, cloud[:, i].min(), cloud[:, i].max())
    points_np = point_cloud
    mask_all = np.ones(points_np.shape[0])
    final_sample_points = None
    mask_ground = None
    for i in range(5):
    # while mask_ground is None:
        best_len = 0
        for iteration in range(min(cloud.shape[0], 100)):
            sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
            
            while check_parallel(sampledPoints) == True:
                sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                continue

            plane = fitPlane(sampledPoints)
            diff = np.abs(np.matmul(points_np, plane) - np.ones(points_np.shape[0])) / np.linalg.norm(plane)
            inlierMask = diff < planeDiffThreshold
            numInliers = inlierMask.sum()
            if numInliers > best_len and np.abs(np.dot(plane/np.linalg.norm(plane),np.array([0,1,0])))>0.9:
                mask_ground = inlierMask
                best_len = numInliers
                best_plane = plane
                final_sample_points = sampledPoints

        if mask_ground is not None:
            mask_all *= 1 - mask_ground
    return mask_all, final_sample_points

def region_grow_my_version(pc, mask_search, mask_origin, thresh, ratio=0.8):
    pc_search = pc[mask_search==1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]))
    while mask.sum() > 0:
        seed = pc[mask==1][0]
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]))
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask==1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search==1] = seed_mask
            if ratio is not None and (seed_mask_all*mask_origin).sum()/seed_mask.sum().astype(np.float32)<ratio:
                flag = 0
                break
        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
        mask *= (1 - seed_mask_all)

    if ratio is not None:
        return mask_best*mask_origin
    else:
        return mask_best

def check_truncate(img_shape, box, threshold=2):
    if min(box[0], box[1]) < 1 or box[2] > img_shape[1] - 2 or box[3] > img_shape[0] - 2:
        return True
    else:
        return False