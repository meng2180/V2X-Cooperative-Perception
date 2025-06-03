import torch
import numpy as np
import torch.nn as nn
from opencood.pcdet_utils.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from opencood.utils import common_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils import box_utils
from icecream import ic
from copy import deepcopy

class VoxelRCNNHead(nn.Module):
    def __init__(self, model_cfg, backbone_channels):    
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = model_cfg['voxel_size']
        self.pool_cfg = model_cfg['pool_cfg']
        self.point_cloud_range = model_cfg['pc_range']
        self.grid_size = self.pool_cfg['grid_size'] # 6
        self.feature_source = self.pool_cfg['feature_source']
        self.code_size = 7

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        
        for src_name in self.feature_source: # FEATURES_SOURCE: ['x_conv2', 'x_conv3', 'x_conv4']
            layer_cfg = self.pool_cfg['pool_layers'][src_name]
            mlps = deepcopy(layer_cfg['mlps'])

            for k in range(len(mlps)): # MLPS: [[32, 32]]
                # backbone_channels: {'x_conv1':16, 'x_conv2':32, 'x_conv3':64, 'x_conv4':64}
                mlps[k] = [backbone_channels[src_name]] + mlps[k]

            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=layer_cfg['query_ranges'],
                nsamples=layer_cfg['nsample'],
                radii=layer_cfg['pool_radius'],
                mlps=mlps,
                pool_method=layer_cfg['pool_method'],
            )
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        # c_out = sum([x[-1] for x in mlps])
        pre_channel = self.grid_size * self.grid_size * self.grid_size * c_out # 20736=6*6*6*96


        fc_layers = [self.model_cfg['n_fc_neurons']] * 2
        self.shared_fc_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                                  fc_layers)

        self.cls_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                            fc_layers,
                                                            output_channels=1)
        self.iou_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=1)
        self.reg_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=7)
        self._init_weights(weight_init='xavier')


    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
                    
    
    def _make_fc_layers(self, input_channels, fc_list, output_channels=None):
        fc_layers = []
        pre_channel = input_channels
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                # nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg['dp_ratio'] > 0:
                fc_layers.append(nn.Dropout(self.model_cfg['dp_ratio']))
        if output_channels is not None:
            fc_layers.append(
                nn.Conv1d(pre_channel, output_channels, kernel_size=1,
                          bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers, pre_channel

    def roi_grid_pool(self, batch_dict):
        """
        roi_grid_pooling happens after box fusion and voxel feature merges

        Args:
            batch_dict:
                batch_size:
                rois: (sum(rois), 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
 
        batch_size = len(batch_dict['record_len'])
        rois = batch_dict['rcnn_label_dict']['rois'] # already lwh order 
        label_record_len = batch_dict['rcnn_label_dict']['record_len']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False) # False
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  # (BxN, 6x6x6, 3) --> (1024, 216, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(-1, 3)  # (sum(proposal)*6*6*6, 3)

        # 2.compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div((roi_grid_xyz[:, 0:1] - self.point_cloud_range[0]), self.voxel_size[0], rounding_mode='floor')
        roi_grid_coords_y = torch.div((roi_grid_xyz[:, 1:2] - self.point_cloud_range[1]), self.voxel_size[1], rounding_mode='floor')
        roi_grid_coords_z = torch.div((roi_grid_xyz[:, 2:3] - self.point_cloud_range[2]), self.voxel_size[2], rounding_mode='floor')

        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        # 3.batch index
        batch_idx = rois.new_zeros(roi_grid_coords.shape[0], 1) 
        idx_start = 0
        for bs_idx in range(batch_size):
            batch_idx[idx_start:idx_start+label_record_len[bs_idx] * self.grid_size ** 3] = bs_idx
            idx_start += label_record_len[bs_idx] * self.grid_size ** 3
        
        # 4.roi grid
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            roi_grid_batch_cnt[bs_idx] = label_record_len[bs_idx] * self.grid_size ** 3

        pooled_features_list = []
        for k, src_name in enumerate(self.feature_source):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features_fused'][src_name]

            # 1.compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride, 
                voxel_size=self.voxel_size, 
                point_cloud_range=self.point_cloud_range 
            ) 
            
            # 2.Count the effective coordinates of each frame of point cloud
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            
            # 3.get voxel2point tensor 
            # Computing the index between voxel coordinates and voxel features
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors) # (8, 21, 800, 704)
            
            # 4.compute the grid coordinates in this scale, in [batch_idx, x y z] order
            cur_roi_grid_coords = torch.div(roi_grid_coords, cur_stride, rounding_mode='floor')
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1) 
            cur_roi_grid_coords = cur_roi_grid_coords.int()

            
            # ic(cur_voxel_xyz.contiguous())
            # ic(cur_voxel_xyz.contiguous().shape)
            # ic(cur_voxel_xyz_batch_cnt)

            # ic(roi_grid_xyz.contiguous().view(-1, 3))
            # ic(roi_grid_xyz.contiguous().view(-1, 3).shape)
            # ic(roi_grid_batch_cnt)

            # ic(cur_roi_grid_coords.contiguous().view(-1, 4))
            # ic(cur_roi_grid_coords.contiguous().view(-1, 4).shape)
            # ic(cur_sp_tensors.features.contiguous())
            # ic(v2p_ind_tensor)
            # ic("___________")


            # 5.voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(), 
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt, 
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3), 
                new_xyz_batch_cnt=roi_grid_batch_cnt, 
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4), 
                features=cur_sp_tensors.features.contiguous(), 
                voxel2point_indices=v2p_ind_tensor
            )


            # 6.Change feature dimensions and add pooled feature list
            pooled_features = pooled_features.view(
                -1, self.grid_size ** 3,
                pooled_features.shape[-1]
            )  # (sum(rcnn_proposal), 6x6x6, C) --> (1024, 216, 32)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features # (sum(rcnn_proposal), 6x6x6, C) --> (1024, 216, 32)


    def get_global_grid_points_of_roi(self, rois, grid_size):
        """
        Args:
            rois:(1024, 7)
            grid_size:6
        Returns:
            global_roi_grid_points, local_roi_grid_points: (1024, 216, 3)
        """
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0] 

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3) --> (1024, 216, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1) # (1024, 216, 3)
        global_center = rois[:, 0:3].clone() # (1024,3)
        global_roi_grid_points += global_center.unsqueeze(dim=1) #  (1024, 216, 3)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        """
        Args:
            rois:(1024, 7)
            batch_size_rcnn:1024
            grid_size:6
        Returns:
            roi_grid_points: (1024, 216, 3)
        """
        faked_features = rois.new_ones((grid_size, grid_size, grid_size)) # Initialize an all 1 6x6x6 pseudo feature
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx] --> (216,3)
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3) --> (1024, 216, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6] # (1024,3)
        # (1024,216,3) / (1024,1,3) - (1024,1,3)
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3) 
        return roi_grid_points # (1024, 216, 3)


    def forward(self, batch_dict):
        batch_dict = self.assign_targets(batch_dict)
        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, self.grid_size,
                              self.grid_size,
                              self.grid_size)  # (BxN, C, 6, 6, 6)
        
        shared_features = self.shared_fc_layers(
            pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1,2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_iou = self.iou_layers(shared_features).transpose(1,2).contiguous().squeeze(dim=1)  # (B, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1,2).contiguous().squeeze(dim=1)  # (B, C)

        batch_dict['stage2_out'] = {
            'rcnn_cls': rcnn_cls,
            'rcnn_iou': rcnn_iou,
            'rcnn_reg': rcnn_reg,
        }

        return batch_dict

    def assign_targets(self, batch_dict):
        batch_dict['rcnn_label_dict'] = {
            'rois': [],
            'gt_of_rois': [],
            'gt_of_rois_src': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
            'rois_anchor': [],
            'record_len': [],
            'rois_scores_stage1': []
        }
        pred_boxes = batch_dict['boxes_fused']
        pred_scores = batch_dict['scores_fused']
        gt_boxes = [b[m][:, [0, 1, 2, 5, 4, 3, 6]].float() for b, m in
                    zip(batch_dict['object_bbx_center'],
                        batch_dict['object_bbx_mask'].bool())]  # hwl -> lwh order
        for rois, scores, gts in zip(pred_boxes, pred_scores,  gt_boxes): # each frame
            rois = rois[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
            if gts.shape[0] == 0:
                gts = rois.clone()

            ious = boxes_iou3d_gpu(rois, gts)
            max_ious, gt_inds = ious.max(dim=1)
            gt_of_rois = gts[gt_inds]
            rcnn_labels = (max_ious > 0.3).float()
            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]
            # TODO: roi_ry > 0 in pcdet
            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * np.pi), rounding_mode='trunc')
                    + 1) * 2 * np.pi) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                    heading_label < np.pi * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + np.pi) % (
                                                   2 * np.pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[
                                      flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2,
                                        max=np.pi / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets = box_utils.box_encode(
                gt_of_rois.view(-1, self.code_size), rois_anchor
            )

            batch_dict['rcnn_label_dict']['rois'].append(rois)
            batch_dict['rcnn_label_dict']['rois_scores_stage1'].append(scores.flatten())
            batch_dict['rcnn_label_dict']['gt_of_rois'].append(gt_of_rois)
            batch_dict['rcnn_label_dict']['gt_of_rois_src'].append(
                gt_of_rois_src)
            batch_dict['rcnn_label_dict']['cls_tgt'].append(rcnn_labels)
            batch_dict['rcnn_label_dict']['reg_tgt'].append(reg_targets)
            batch_dict['rcnn_label_dict']['iou_tgt'].append(max_ious)
            batch_dict['rcnn_label_dict']['rois_anchor'].append(rois_anchor)
            batch_dict['rcnn_label_dict']['record_len'].append(rois.shape[0])
            

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict'][k] = torch.cat(v, dim=0)

        return batch_dict