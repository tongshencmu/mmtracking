
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmdet.structures.bbox.transforms import bbox_xyxy_to_cxcywh
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

from mmtrack.registry import MODELS
from mmtrack.structures import TrackDataSample
from mmtrack.utils import (InstanceList, OptConfigType, OptMultiConfig,
                           SampleList)
from .base import BaseSingleObjectTracker


@MODELS.register_module()
class VLTracker(BaseSingleObjectTracker):
    
    def __init__(self, 
                 backbone, 
                 cls_head,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 frozen_modules = None,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor, init_cfg)
        
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)
        self.bbox_head = MODELS.build(bbox_head)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)
            
    def extract_feat(self, img: Tensor) -> Tensor:
        """Extract the features of the input image.

        Args:
            img (Tensor): image of shape (N, C, H, W).

        Returns:
            tuple(Tensor): the multi-level feature maps, and each of them is
                    of shape (N, C, H // stride, W // stride).
        """
        feat = self.backbone(img)
        return feat
    
    def fuse_feature(self, feat: Tensor) -> Tensor:
        """Fuse the features of the input image.

        Args:
            x_feat (Tensor): feature of search area in shape (N, C, H, W).
            z_feat (Tensor): feature of template area in shape (N, C, H, W).

        Returns:
            Tensor: the fused feature map of shape (N, C, H, W).
        """
        feat = self.cls_head(feat)
        return feat
            
    def forward_train(self, inputs, data_samples, **kwargs):
                      
        """Forward of training.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                template/search frames.
                - img (Tensor) : The template images.
                - search_img (Tensor): The search images.
                - text (Tensor): The text prompts.
                
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Return:
            dict: A dictionary of loss components.
        """
        
        search_img = inputs['search_img']
        assert search_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        search_img = search_img[:, 0]

        template_img = inputs['img']
        assert template_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        template_img = template_img[:, 0]
        
        text = inputs['text']
        
        # extract feature
        x_feat = self.extract_feat(search_img)
        z_feat = self.extract_feat(template_img)
        
        # fuse feature
        fused_feat, pred_token = self.fuse_feature(x_feat)
        
        # Run box head for regression
        
        
        