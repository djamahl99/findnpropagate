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
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
import random

from ..model_utils import model_nms_utils
import time
import cv2
from ..preprocessed_detector import PreprocessedDetector, PreprocessedGLIP

from sklearn.decomposition import PCA

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[2,2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
    return rotmat

def compute_bbox(in_pc):
    pca = PCA(2)
    pca.fit(in_pc[:,:2])
    yaw_vec = pca.components_[0,:]
    yaw = np.arctan2(yaw_vec[1],yaw_vec[0])

    in_pc_tmp = in_pc.copy()
    in_pc_tmp = heading2rotmat(-yaw) @ in_pc_tmp[:,:3].T

    x_min = in_pc_tmp[0,:].min()
    x_max = in_pc_tmp[0,:].max()
    y_min = in_pc_tmp[1,:].min()
    y_max = in_pc_tmp[1,:].max()
    z_min = in_pc_tmp[2,:].min()
    z_max = in_pc_tmp[2,:].max()

    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min

    bbox = heading2rotmat(yaw) @ np.array([[x_min,y_min,z_min],[x_max,y_max,z_max]]).T
    bbox = bbox.T
    x_min,y_min,z_min = bbox[0]
    x_max,y_max,z_max = bbox[1]

    cx = (x_min+x_max)/2
    cy = (y_min+y_max)/2
    cz = (z_min+z_max)/2

    # rst_bbox = np.expand_dims(np.array([cx, cy, cz, dx/2, dy/2, dz/2, -1*yaw]), axis=0)
    #print(rst_bbox.shape)


    #write_oriented_bbox(rst_bbox, "rst.ply")
    #write_ply(in_pc[:,:3], "pc.ply")
    #print(cx, cy, cz, dx, dy, dz, yaw)
    #exit()
    # return cx, cy, cz, dx/2, dy/2, dz/2, -1*yaw
    return cx, cy, cz, dx, dy, dz, -1*yaw

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

class FrustumOV3DET(nn.Module):
    def __init__(
        self,
        model_cfg=None, input_channels=None, num_class=None, class_names=None, grid_size=None, point_cloud_range=None, voxel_size=None, predict_boxes_when_training=True,
    ):
        super(FrustumOV3DET, self).__init__()


        # self.cluster_eps = 1.5 # 
        self.score_thr = model_cfg.SCORE_THRESH


        self.image_order = [2, 0, 1, 5, 3, 4]
        # self.image_size = [512, 800]
        self.image_size = [900, 1600]

        self.point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

        self.frustum_min = torch.tensor(2.0, device='cuda')
        self.frustum_max = torch.tensor(60, device='cuda')

        self.box_fmt = model_cfg.get('BOX_FORMAT', 'xyxy')
        preds_path = model_cfg.get('PREDS_PATH', '/home/uqdetche/GLIP/jsons/OWL_')
        self.nms_2d = 0.6

        print('preds_path', preds_path)
        if 'PreprocessedGLIP' in preds_path:
            self.image_detector = PreprocessedGLIP(class_names=class_names)
        else:
            # exit()
            # preds_path = '/home/uqdetche/OpenPCDet/tools/coco_val2_'
            # preds_path = '/home/uqdetche/GLIP/OWL_'
            # preds_path = '/home/uqdetche/GLIP/jsons/OWL_'
            camera_names = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
            preds_paths = [preds_path + f"{cam_name}.json" for cam_name in camera_names]

            preds_paths = model_cfg.get('PREDS_PATHS', preds_paths)
            print('preds_paths', preds_paths)
            self.image_detector = PreprocessedDetector(preds_paths, class_names=class_names)

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

    def get_proposals(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        # images = batch_dict['camera_imgs']

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

        proposal_boxes = []
        proposal_scores = []
        frust_labels = []
        frust_batch_idx = []
        frust_scores = []
        frust_indices = []

        for b in range(batch_size):
            cur_points = batch_dict['points'][batch_dict['points'][..., 0] == b, 1:4]

            detector_batch_mask = (det_batch_idx == b)
            cur_boxes, cur_labels, cur_scores, cur_cam_idx = det_boxes[detector_batch_mask], det_labels[detector_batch_mask], det_scores[detector_batch_mask], det_cam_idx[detector_batch_mask]

            for c in self.image_order:
                box_cam_mask = (cur_cam_idx == c)
                cam_boxes, cam_labels, cam_scores = cur_boxes[box_cam_mask], cur_labels[box_cam_mask], cur_scores[box_cam_mask]

                if cam_boxes.shape[0] > 0:
                    selected = batched_nms(cam_boxes, cam_scores, cam_labels, self.nms_2d)
                    cam_boxes, cam_labels, cam_scores = cam_boxes[selected], cam_labels[selected], cam_scores[selected]

                cam_points, cam_mask = self.project_to_camera(batch_dict, cur_points, batch_idx=b, cam_idx=c)
                cam_points = cam_points[cam_mask]

                cam_points_3d = cur_points[cam_mask[0]]

                for box, label, score in zip(cam_boxes, cam_labels, cam_scores):
                    if score < self.score_thr:
                        continue

                    if self.box_fmt == 'xyxy':
                        x1, y1, x2, y2 = box.cpu()
                    else:
                        box[..., 2:] += box[..., 0:2]
                        x1, y1, x2, y2 = box

                    # if (x2 - x1) * (y2 - y1) > (1600 * 900) *0.5:
                        # continue

                    on_box = (
                        (cam_points[..., 1] < y2)
                        & (cam_points[..., 1] >= y1)
                        & (cam_points[..., 0] < x2)
                        & (cam_points[..., 0] >= x1)
                    )

                    if on_box.sum() == 0:
                        continue

                    frustum_points = cam_points_3d[on_box].reshape(-1, 3).detach().cpu().numpy()

                    # below modified from https://github.com/lyhdet/OV-3DET/blob/main/Data_Maker/Pseudo_Label_Maker/scannet_util.py#L329
                    cur_ins_pc = frustum_points.copy()

                    step_interval = max((1, int(cur_ins_pc.shape[0]/3000)))
                    cur_ins_pc = cur_ins_pc[0:cur_ins_pc.shape[0]:step_interval, :]
                    
                    if cur_ins_pc.shape[0] < 100:
                        continue

                    db = DBSCAN(eps=0.75, min_samples=20, metric = 'euclidean').fit(cur_ins_pc)

                    cur_ins_pc_remove_outiler = []
                    for cluster in np.unique(db.labels_):
                        if cluster < 0:
                            continue
                        cluster_ind = db.labels_ == cluster

                        cur_ins_pc_remove_outiler.append(cur_ins_pc[cluster_ind,:])

                    if len(cur_ins_pc_remove_outiler) < 1:
                        continue

                    cur_ins_pc = np.concatenate(cur_ins_pc_remove_outiler, axis=0)

                    xc, yc, zc, l, w, h, yaw = compute_bbox(cur_ins_pc)
                    # above modified from https://github.com/lyhdet/OV-3DET/blob/main/Data_Maker/Pseudo_Label_Maker/scannet_util.py#L329


                    proposal_boxes.append(torch.tensor([xc, yc, zc, l, w, h, -1*yaw], dtype=torch.float32).reshape(1, 7))
                    frust_labels.append(label)
                    frust_scores.append(score)
                    frust_batch_idx.append(b)


        if len(proposal_boxes) > 0:
            proposal_boxes = torch.cat(proposal_boxes, dim=0).reshape(-1, 7).to('cuda')
            frust_labels = torch.tensor(frust_labels, dtype=torch.long)
            frust_scores = torch.tensor(frust_scores)
            frust_batch_idx = torch.tensor(frust_batch_idx, dtype=torch.long)
            assert frust_scores.shape[0] == frust_labels.shape[0] and proposal_boxes.shape[0] == frust_scores.shape[0], \
                f"boxes, labels, scores shapes {proposal_boxes.shape}, {frust_labels.shape}, {frust_scores.shape}"
        else:
            proposal_boxes = torch.zeros((0, 7), device='cuda')
            frust_labels = torch.zeros((0), dtype=torch.long)
            frust_scores = torch.zeros((0))
            frust_batch_idx = torch.zeros((0), dtype=torch.long)

        return proposal_boxes, frust_labels, frust_scores, frust_batch_idx

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

            # ret_dict.append()
            # print('boxes', proposed_boxes[mask].shape)
            # print('scores', proposed_scores[mask].shape)
            # print('labels', proposed_labels[mask].shape)

            ret_dict[k]['pred_boxes'] = proposed_boxes[mask]#.cpu()
            ret_dict[k]['pred_scores'] = proposed_scores[mask]
            ret_dict[k]['pred_labels'] = proposed_labels[mask].int() # + 1 is done in preprocessed_detector

        return ret_dict 

def fit_plane(data: np.ndarray, max_iterations=5, sample_size=10, threshold=0.1, goal_inliers_p=0.5):

    # number of points we want to sample, 3 is minimum, but we can choose more for better fitting.
    # sample_size = 10
    # distance threshold for choosing inliers
    # threshold = 0.1
    # minimum numbers of inliers we need to have, we can ignore this parameter by setting None
    goal_inliers = data.shape[0] * goal_inliers_p

    coeff, _ = ransac(data[:, :3], model, lambda x, y: is_inlier(x, y, threshold), sample_size, max_iterations, goal_inliers)
    proj = data[:,0] * coeff[0] + data[:,1] * coeff[1] + data[:,2] * coeff[2] + coeff[3]

    ground_pts_mask = np.abs(proj) <= threshold
    
    return ground_pts_mask

def ransac(data, model, is_inlier, sample_size, max_iterations, goal_inliers, random_seed=75, debug=False):
    best_ic = 0
    best_model = None
    random.seed(random_seed)

    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = model(s)
        ic = 0
        for j in range(len(data)):
              if is_inlier(m, data[j]):
                      ic += 1

        if debug:
              print('Coeffs:', m, '# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if goal_inliers and ic > goal_inliers:
                  break
    if debug: 
          print('Took iterations:', i+1, 'Best model coeffs:', best_model, 'Inliers covered:', best_ic)

    return best_model, best_ic

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def model(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold