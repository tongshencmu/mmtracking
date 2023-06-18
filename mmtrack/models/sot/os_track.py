# Copyright (c) OpenMMLab. All rights reserved.
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
                           SampleList, tokenize)
from .base import BaseSingleObjectTracker

import math

@MODELS.register_module()
class OSTrack(BaseSingleObjectTracker):
    
    """
    Args:
        backbone (dict): the configuration of backbone network.
        neck (dict, optional): the configuration of neck network.
            Defaults to None.
        head (dict, optional): the configuration of head network.
            Defaults to None.
        init_cfg (dict, optional): the configuration of initialization.
            Defaults to None.
        frozen_modules (str | list | tuple, optional): the names of frozen
            modules. Defaults to None.
        train_cfg (dict, optional): the configuratioin of train.
            Defaults to None.
        test_cfg (dict, optional): the configuration of test.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrains: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 frozen_modules: Optional[Union[List[str], Tuple[str],
                                                str]] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(OSTrack, self).__init__(data_preprocessor, init_cfg)
        # head.update(test_cfg=test_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.head = MODELS.build(head)
        
        self.head_feat_sz = self.head.feat_sz

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.num_templates = 1

        # Set the update interval
        self.update_intervals = self.test_cfg.get('update_intervals', None)
        if isinstance(self.update_intervals, (int, float)):
            self.update_intervals = [int(self.update_intervals)
                                     ] * self.num_templates

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)
            
    def init_weights(self):
        """Initialize the weights of modules in single object tracker."""
        # We don't use the `init_weights()` function in BaseModule, since it
        # doesn't support the initialization method from `reset_parameters()`
        # in Pytorch.
        if self.with_backbone:
            self.backbone.init_weights()

        if self.with_neck:
            for m in self.neck.modules():
                if isinstance(m, _ConvNd) or isinstance(m, _BatchNorm):
                    m.reset_parameters()

        if self.with_head:
            self.head.init_weights()
            
    def extract_feat(self, img: Tensor) -> Tensor:
        """Extract the features of the input image.

        Args:
            img (Tensor): image of shape (N, C, H, W).

        Returns:
            tuple(Tensor): the multi-level feature maps, and each of them is
                    of shape (N, C, H // stride, W // stride).
        """
        feat = self.backbone(img)
        width = feat.shape[1]
        H = int(math.sqrt(width - 1))
        size = feat.shape
        feat = [feat[:, 1:, :].permute(0, 2, 1).reshape(size[0], size[2], H, H)]
        feat = self.neck(feat)
        return feat
    
    def predict_vot(self, inputs: dict, data_samples: List[TrackDataSample]):
        raise NotImplementedError(
            'STARK does not support testing on VOT datasets yet.')

    def loss(self, inputs: dict, data_samples: List[TrackDataSample],
             **kwargs) -> dict:
        """Forward of training.

        Args:
            inputs (dict[Tensor]): of shape (N, T, C, H, W) encoding
                input images. Typically these should be mean centered and std
                scaled. The N denotes batch size. The T denotes the number of
                key/reference frames.
                - img (Tensor) : The key images.
                - ref_img (Tensor): The reference images.

            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.

        Return:
            dict: A dictionary of loss components.
        """
        # template_padding_mask = [
        #     data_sample.padding_mask for data_sample in data_samples
        # ]
        # template_padding_mask = torch.stack(template_padding_mask, dim=0)
        # search_padding_mask = [
        #     data_sample.search_padding_mask for data_sample in data_samples
        # ]
        # search_padding_mask = torch.stack(search_padding_mask, dim=0)

        search_img = inputs['search_img']
        assert search_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        template_img = inputs['img']
        assert template_img.dim(
        ) == 5, 'The img must be 5D Tensor (N, T, C, H, W).'

        feat = self.backbone(search_img, template_img)
        search_feat = feat[0, 1:(self.head_feat_sz * self.head_feat_sz +1), :].permute(0, 2, 1)
        search_feat = search_feat.reshape(search_feat.shape[0], -1, self.head_feat_sz, self.head_feat_sz)

        losses = self.head.loss(search_feat, data_samples)

        return losses
