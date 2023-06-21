
import math
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
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
                 head,
                 train_cfg,
                 test_cfg,
                 frozen_modules = None,
                 data_preprocessor: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None) -> None:
        super(VLTracker, self).__init__(data_preprocessor, init_cfg)
        
        self.backbone = MODELS.build(backbone)
        for name, param in self.backbone.named_parameters():
            print(name)
        self.head = MODELS.build(head)
        
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
    
    def fuse_feature(self, x_feat: Tensor, z_feat: Tensor) -> Tensor:
        """Fuse the features of the input image.

        Args:
            x_feat (Tensor): feature of search area in shape (N, C, H, W).
            z_feat (Tensor): feature of template area in shape (N, C, H, W).

        Returns:
            Tensor: the fused feature map of shape (N, C, H, W).
        """
        x_emb, z_emb, token_emb = self.feat_fusion(x_feat, z_feat)
        return x_emb, z_emb, token_emb
    
    def forward_bbox_head(self, feat: Tensor, enc_mem: Tensor) -> Tensor:
        """
        Args:
            feat (Tensor): output embeddings of decoder, with shape
                (1, bs, num_query, c).
            enc_mem (Tensor): output embeddings of encoder, with shape
                (feats_flatten_len, bs, C)

                Here, 'feats_flatten_len' = z_feat_h*z_feat_w*2 + \
                    x_feat_h*x_feat_w.
                'z_feat_h' and 'z_feat_w' denote the height and width of the
                template features respectively.
                'x_feat_h' and 'x_feat_w' denote the height and width of search
                features respectively.
        Returns:
            Tensor: of shape (bs * num_query, 4). The bbox is in
                [tl_x, tl_y, br_x, br_y] format.
        """
        z_feat_len = self.bbox_head.feat_size**2
        # the output of encoder for the search image
        x_feat = enc_mem[-z_feat_len:].transpose(
            0, 1)  # (bs, x_feat_h*x_feat_w, c)
        dec_embed = feat.squeeze(0).transpose(1, 2)  # (bs, c, num_query)
        attention = torch.matmul(
            x_feat, dec_embed)  # (bs, x_feat_h*x_feat_w, num_query)
        bbox_feat = (x_feat.unsqueeze(-1) * attention.unsqueeze(-2))

        # (bs, x_feat_h*x_feat_w, c, num_query) --> (bs, num_query, c, x_feat_h*x_feat_w) # noqa
        bbox_feat = bbox_feat.permute((0, 3, 2, 1)).contiguous()
        bs, num_query, dim, _ = bbox_feat.size()
        bbox_feat = bbox_feat.view(-1, dim, self.bbox_head.feat_size,
                                   self.bbox_head.feat_size)
        # run the corner prediction head
        outputs_coord = self.bbox_head(bbox_feat)
        return outputs_coord
            
    def loss(self, inputs, data_samples, **kwargs):
                      
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
        
        # text = inputs['text']
        
        # extract feature
        x_feat = self.extract_feat(search_img)     # (N, L, D)
        z_feat = self.extract_feat(template_img)   # (N, L, D)
        
        x_feat = x_feat[:, 1:, :]  
        z_feat = z_feat[:, 1:, :]
        
        head_inputs = dict(x_embeddings=x_feat, z_embeddings=z_feat)
        # head_inputs.append(dict(x_embeddings=x_feat))
        # head_inputs.append(dict(z_embeddings=z_feat))
        
        loss = self.head.loss(head_inputs, data_samples)

        return loss
        
    def init(img: Tensor):
        """Initialize the single object tracker in the first frame.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
        """
        pass

    def track(img: Tensor, data_samples: SampleList) -> InstanceList:
        """Track the box of previous frame to current frame `img`.

        Args:
            img (Tensor): of shape (1, C, H, W) encoding original input
                image.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as ``gt_instances`` and 'metainfo'.

        Returns:
            InstanceList: Tracking results of each image after the postprocess.
                - scores: a Tensor denoting the score of best_bbox.
                - bboxes: a Tensor of shape (4, ) in [x1, x2, y1, y2]
                format, and denotes the best tracked bbox in current frame.
        """
        pass
        