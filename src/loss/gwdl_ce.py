# Copyright 2021 Lucas Fidon and Suprosanna Shit

import torch
import torch.nn as nn
import numpy as np
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss


# We assume the BraTS classes are in this order
# 1: Non-enhancing Tumor (NET)
# 2: Edema
# 3: Enhancing Tumor (ET)
# dist(ET, NET) = 0.5
# dist(edema, NET) = 0.6
# dist(edema, ET) = 0.7
BRATS_DIST_MATRIX = np.array(
    [[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.6, 0.5],
    [1.0, 0.6, 0.0, 0.7],
    [1.0, 0.5, 0.7, 0.0]],
    dtype=np.float64
)
# Variation that makes more sense with the class hierarchy
# dist(ET, NET) = 0.5
# dist(edema, NET) = 0.7
# dist(edema, ET) = 0.6
BRATS_DIST_MATRIXv2 = np.array(
    [[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.7, 0.5],
    [1.0, 0.7, 0.0, 0.6],
    [1.0, 0.5, 0.6, 0.0]],
    dtype=np.float64
)

class GWDLCELoss(nn.Module):
    """Generalized Wasserstein Dice loss + Xentropy loss"""
    def __init__(self, dist_matrix, reduction='mean'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gwdl = GeneralizedWassersteinDiceLoss(dist_matrix=dist_matrix)
        self.ce.reduction = reduction
        self.gwdl.reduction = reduction

    def forward(self, y_pred, y_true):
        gwdl_loss = self.gwdl(y_pred, y_true.long())
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        ce_loss = self.ce(y_pred, torch.squeeze(y_true, dim=1).long())
        result = ce_loss + gwdl_loss
        return result

    @property
    def reduction(self):
        return self.ce.reduction  # should be the same as self.gwdl.reduction

    @reduction.setter
    def reduction(self, reduction):
        self.ce.reduction = reduction
        self.gwdl.reduction = reduction


class GWDLCELossBraTS(GWDLCELoss):
    def __init__(self, reduction='mean'):
        super(GWDLCELossBraTS, self).__init__(
            dist_matrix=BRATS_DIST_MATRIX,
            reduction=reduction,
        )

class GWDLCELossBraTSv2(GWDLCELoss):
    def __init__(self, reduction='mean'):
        super(GWDLCELossBraTSv2, self).__init__(
            dist_matrix=BRATS_DIST_MATRIXv2,
            reduction=reduction,
        )
