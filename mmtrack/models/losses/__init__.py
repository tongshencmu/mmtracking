# Copyright (c) OpenMMLab. All rights reserved.
from .kl_loss import KLGridLoss, KLMCLoss
from .l2_loss import L2Loss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss
from .triplet_loss import TripletLoss
from .focal_loss import FocalLoss
from .hinge_loss import LBHinge

__all__ = [
    'TripletLoss', 'MultiPosCrossEntropyLoss', 'L2Loss', 'KLGridLoss',
    'KLMCLoss', 'FocalLoss', 'LBHinge'
]
