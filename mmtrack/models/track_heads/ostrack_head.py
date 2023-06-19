# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.layers import Transformer
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmtrack.registry import MODELS
from mmtrack.utils import InstanceList, OptConfigType, SampleList, generate_heatmap

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

@MODELS.register_module()
class CenterPredictHead(BaseModule):
    
    def __init__(self, 
                 inplanes=64, 
                 channel=256, 
                 feat_sz=20, 
                 stride=16, 
                 freeze_bn=False):
        
        super(CenterPredictHead, self).__init__()
        
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        
        def conv_module(in_planes: int,
                        out_planes: int,
                        kernel_size: int = 3,
                        padding: int = 1):
            # The module's pipeline: Conv -> BN -> ReLU.
            return ConvModule(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='ReLU'),
                inplace=True)
        
        # center predictor
        self.score_pred = nn.Sequential(
            conv_module(inplanes, channel), conv_module(channel, channel // 2),
            conv_module(channel // 2, channel // 4),
            conv_module(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 1, kernel_size=1))
        
        self.offset_pred = nn.Sequential(
            conv_module(inplanes, channel), conv_module(channel, channel // 2),
            conv_module(channel // 2, channel // 4),
            conv_module(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 2, kernel_size=1))
        
        self.size_pred = nn.Sequential(
            conv_module(inplanes, channel), conv_module(channel, channel // 2),
            conv_module(channel // 2, channel // 4),
            conv_module(channel // 4, channel // 8),
            nn.Conv2d(channel // 8, 2, kernel_size=1))
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        
        score_map_ctr, score_map_size, score_map_offset = self.get_score_map(x)
        
        bbox_xyxy = self.cal_bbox(score_map_ctr, score_map_size, score_map_offset)
        
        return score_map_ctr, bbox_xyxy
    
    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox_cxcywh = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)
        
        bbox_xyxy = bbox_cxcywh_to_xyxy(bbox_cxcywh)

        if return_score:
            return bbox_xyxy, max_score
        return bbox_xyxy
    
    def get_score_map(self, x):
        
        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        score_map_ctr = self.score_pred(x)

        # offset branch
        score_map_offset = self.offset_pred(x)

        # size branch
        score_map_size = self.size_pred(x)
        
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset
    
    
@MODELS.register_module()
class OSTrackHead(BaseModule):
    
    def __init__(self, 
                 bbox_head=None,
                 feat_sz=20,
                 loss_cls=None,
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 frozen_modules=None,
                 **kwargs):
        
        """
        Args:
        bbox_head (obj:`mmengine.ConfigDict`|dict, optional): Config for bbox
            head. Defaults to None.
        loss_cls (obj:`mmengine.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmengine.ConfigDict`|dict): Config of the bbox
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmengine.ConfigDict`|dict): Config of the bbox
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmengine.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmengine.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    
        super(OSTrackHead, self).__init__()
        
        assert bbox_head is not None
        
        self.feat_sz = feat_sz
        self.bbox_head = MODELS.build(bbox_head)

        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_focal = MODELS.build(loss_cls)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        if frozen_modules is not None:
            assert isinstance(frozen_modules, list)
            for module in frozen_modules:
                m = getattr(self, module)
                # TODO: Study the influence of freezing BN running_mean and
                # running_variance of `frozen_modules` in the 2nd stage train.
                # The official code doesn't freeze these.
                for param in m.parameters():
                    param.requires_grad = False
                    
    def init_weights(self):
        """Parameters initialization."""
        self.bbox_head.init_weights()
        
    def forward(self, x: Tensor):

        score_map_ctr, bbox = self.bbox_head(x)
        
        return score_map_ctr, bbox
    
    def loss(self, inputs: Tensor, data_samples: SampleList,
             **kwargs) -> dict:
        """Compute loss.

        Args:
            inputs (list[dict(tuple(Tensor))]): The list contains the
                multi-level features and masks of template or search images.
                    - 'feat': (tuple(Tensor)), the Tensor is of shape
                        (bs, c, h//stride, w//stride).
                    - 'mask': (Tensor), of shape (bs, h, w).
                Here, `h` and `w` denote the height and width of input
                image respectively. `stride` is the stride of feature map.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`
                and 'metainfo'.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        outs = self(inputs)

        batch_gt_instances = []
        batch_img_metas = []
        batch_search_gt_instances = []
        for data_sample in data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            batch_search_gt_instances.append(data_sample.search_gt_instances)

        loss_inputs = outs + (batch_gt_instances, batch_search_gt_instances,
                              batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses
    
    def loss_by_feat(self, score_map_ctr: Tensor, pred_bboxes: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_search_gt_instances: InstanceList,
                     batch_img_metas: List[dict]) -> dict:
        """Compute loss.

        Args:
            pred_logits: (Tensor) of shape (bs * num_query, 1). This item
                only exists when the model has classification head.
            pred_bboxes: (Tensor) of shape (bs * num_query, 4), in
                [tl_x, tl_y, br_x, br_y] format
            batch_gt_instances (InstanceList): the instances in a batch.
            batch_search_gt_instances (InstanceList): the search instances in a
                batch.
            batch_img_metas (List[dict]): the meta information of all images in
                a batch.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses = dict()

        assert pred_bboxes is not None
        img_shape = batch_img_metas[0]['search_img_shape']
        # pred_bboxes[:, 0:4:2] = pred_bboxes[:, 0:4:2] / float(img_shape[1])
        # pred_bboxes[:, 1:4:2] = pred_bboxes[:, 1:4:2] / float(img_shape[0])

        gt_bboxes = [
            instance['bboxes'] for instance in batch_search_gt_instances
        ]
        gt_bboxes = torch.cat(gt_bboxes, dim=0).type(torch.float32)
        gt_bboxes[:, 0:4:2] = gt_bboxes[:, 0:4:2] / float(img_shape[1])
        gt_bboxes[:, 1:4:2] = gt_bboxes[:, 1:4:2] / float(img_shape[0])
        gt_bboxes = gt_bboxes.clamp(0., 1.)

        # regression IoU loss, default GIoU loss
        if (pred_bboxes[:, :2] >= pred_bboxes[:, 2:]).any() or (
                gt_bboxes[:, :2] >= gt_bboxes[:, 2:]).any():
            # the first several iterations of train may return invalid
            # bbox coordinates.
            losses['loss_iou'] = (pred_bboxes - gt_bboxes).sum() * 0.0
        else:
            losses['loss_iou'] = self.loss_iou(pred_bboxes, gt_bboxes)
        # regression L1 loss
        losses['loss_bbox'] = self.loss_bbox(pred_bboxes, gt_bboxes)
        
        # Add focal loss to gaussian heatmap
        gt_bboxes_xywh = gt_bboxes.clone()
        gt_bboxes_xywh[:, 2:] = gt_bboxes_xywh[:, 2:] - gt_bboxes_xywh[:, :2]
        
        gt_gaussian_map = generate_heatmap(gt_bboxes_xywh.unsqueeze(1), img_shape[0], self.bbox_head.stride)
        losses['loss_focal'] = self.loss_focal(score_map_ctr, gt_gaussian_map)

        return losses

    