import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import ipdb
from torch.nn import init

from pcdet.utils.frustum_model_util import *
from pcdet.utils import common_utils


class PointNetInstanceSeg(nn.Module):
    def __init__(self,n_classes=3,n_channel=3):
        '''v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1088+n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec): # bs,4,n
        '''
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))# bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0] #bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs,-1,1)#bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs,-1,1)\
                .repeat(1,1,n_pts)# bs,1027,n
        concat_feat = torch.cat([out2,\
            expand_global_feat_repeat],1)
        # bs, (641024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))#bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))#bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))#bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))#bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)#bs, 2, n

        seg_pred = x.transpose(2,1).contiguous()#bs, n, 2
        return seg_pred

class PointNetEstimation(nn.Module):
    def __init__(self,n_classes=2, n_heading_bin:int=12):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes
        self.n_heading_bin = n_heading_bin

        self.fc1 = nn.Linear(512+self.n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+self.n_heading_bin*2+self.n_classes*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts,one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
    def forward(self, pts,one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        return x

class FrustumPointNetv1(nn.Module):
    def __init__(self,n_classes=3,n_channel=3,n_heading_bin=12, hierarchy_anchors=None):
        super(FrustumPointNetv1, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.n_heading_bin = n_heading_bin
        self.InsSeg = PointNetInstanceSeg(n_classes=n_classes,n_channel=n_channel)
        self.STN = STNxyz(n_classes=n_classes)
        self.est = PointNetEstimation(n_classes=n_classes, n_heading_bin=self.n_heading_bin)
        self.Loss = FrustumPointNetLoss()

        self.hierarchy_anchors = hierarchy_anchors

    def forward(self, data_dicts, pred: bool=False):
        point_cloud = data_dicts.get('points')#torch.Size([32, 4, 1024])

        # channel first
        point_cloud = point_cloud.permute(0, 2, 1)

        point_cloud = point_cloud[:,:self.n_channel,:]
        one_hot = data_dicts.get('one_hot')#torch.Size([32, 3])
        bs = point_cloud.shape[0]
        # If not None, use to Compute Loss
        seg_label = data_dicts.get('mask_label')#torch.Size([32, 1024])
        box3d_center_label = data_dicts.get('box3d_center')#torch.Size([32, 3])
        size_class_label = data_dicts.get('size_class')#torch.Size([32, 1])
        size_residual_label = data_dicts.get('size_residual')#torch.Size([32, 3])
        heading_class_label = data_dicts.get('angle_class')#torch.Size([32, 1])
        heading_residual_label = data_dicts.get('angle_residual')#torch.Size([32, 1])

        # print('point_cloud, one_hot', point_cloud.device, one_hot.device)
        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(point_cloud, one_hot)#bs,n,2

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = \
                 point_cloud_masking(point_cloud, logits)

        # T-Net
        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz,one_hot)#(32,3)
        stage1_center = center_delta + mask_xyz_mean#(32,3)

        # if(np.isnan(stage1_center.cpu().detach().numpy()).any()):
            # ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
                    center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new,one_hot)#(32, 59)

        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred, logits, mask, stage1_center, self.n_classes, self.n_heading_bin, self.hierarchy_anchors)

        box3d_center = center_boxnet + stage1_center #bs,3

        if pred:
            # do prediction

            size_scores = torch.softmax(size_scores, dim=-1)
            # calc dim
            print('size_scores', size_scores.shape, 'self.hierarchy_anchors', self.hierarchy_anchors.shape)
            print('size_residual', size_residual.shape)

            hierarchy_anchors = self.hierarchy_anchors.unsqueeze(0).to(size_scores.device)
            size_scores_exp = size_scores.unsqueeze(2)
            dim = (size_scores_exp * hierarchy_anchors).sum(dim=1) + (size_scores_exp * size_residual).sum(dim=1)
            # dim = dim.sum(dim=1, keepdim=True)
            # print('dim', dim.shape)
            # print('dim', dim)
            # print('size scores', size_scores)
            # rot = class2angle(heading_scores, heading_residual, self.n_heading_bin)
            # print('rot', rot.shape)

            print('heading', heading_scores.shape, heading_residual.shape)

            heading_cls = torch.argmax(heading_scores, dim=-1)
            heading_cls_float = heading_cls.float()
            rot = heading_cls_float * (2*np.pi/float(heading_scores.shape[-1])) + (heading_residual * torch.softmax(heading_scores, dim=-1)).sum(dim=-1)

            # add pre rotation
            rot = rot + data_dicts['prerot']

            # rotate centers
            box3d_center = common_utils.rotate_points_along_z(box3d_center.reshape(-1, 1, 3), data_dicts['prerot'])
            box3d_center = box3d_center.reshape(-1, 3)

            return dict(
                heatmap=one_hot,
                center=box3d_center[..., :2],
                height=box3d_center[..., [2]],
                dim=dim,
                rot=rot.unsqueeze(-1))

        losses = self.Loss(logits, seg_label, \
                 box3d_center, box3d_center_label, stage1_center, \
                 heading_scores, heading_residual_normalized, \
                 heading_residual, \
                 heading_class_label, heading_residual_label, \
                 size_scores, size_residual_normalized, \
                 size_residual, \
                 size_class_label, size_residual_label, hierarchy_anchors=self.hierarchy_anchors)

        for key in losses.keys():
            losses[key] = losses[key]/bs

        with torch.no_grad():
            seg_correct = torch.argmax(logits.detach().cpu(), 2).eq(seg_label.detach().cpu()).numpy()
            seg_accuracy = np.sum(seg_correct) / float(point_cloud.shape[-1])

            iou2ds, iou3ds = compute_box3d_iou( \
                box3d_center.detach().cpu().numpy(),
                heading_scores.detach().cpu().numpy(),
                heading_residual.detach().cpu().numpy(),
                size_scores.detach().cpu().numpy(),
                size_residual.detach().cpu().numpy(),
                box3d_center_label.detach().cpu().numpy(),
                heading_class_label.detach().cpu().numpy(),
                heading_residual_label.detach().cpu().numpy(),
                size_class_label.detach().cpu().numpy(),
                size_residual_label.detach().cpu().numpy(), self.hierarchy_anchors.detach().cpu().numpy())
        metrics = {
            'seg_acc': seg_accuracy,
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.7': np.sum(iou3ds >= 0.7)/bs
        }
        return losses, metrics