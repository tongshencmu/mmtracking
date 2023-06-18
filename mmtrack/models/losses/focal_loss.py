# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmtrack.registry import MODELS

@MODELS.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)

        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss