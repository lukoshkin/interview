import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_curve


def ROC_EER(y_true, y_pred, return_roc=False):
    if isinstance(y_pred, torch.Tensor):
        y_true = y_true.cpu()
        y_pred = y_pred.detach().cpu()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    eer = fpr[np.argmin(np.abs(1 - tpr - fpr))]
    if return_roc: return eer, fpr, tpr
    return eer


class LabelSmoothedBCEwLL(nn.BCEWithLogitsLoss):
    """
    In this implementation, the effect of label smoothing is
    taken into account in the regularization term. This loss
    is completely equivalent to the actual label smoothing
    implemented below, when the hyperparmeter epsilon is
    twice as the commented out one.
    """
    def __init__(self, eps=.2):
        super().__init__()
        self.eps = eps
        
    def forward(self, input, target):
        base_loss = super().forward(input, target)
        dummy_target = torch.empty_like(target).fill_(.5)
        reg_term = super().forward(input, dummy_target)
        return (1-self.eps)*base_loss + self.eps*reg_term

# class LabelSmoothedBCEwLL(nn.BCEWithLogitsLoss):
#     def __init__(self, eps=.1):
#         super().__init__()
#         self.eps = eps
        
#     def forward(self, input, target):
#         target = target.clone()
#         target[target == 0] = self.eps
#         target[target == 1] = 1 - self.eps
#         return super().forward(input, target)
