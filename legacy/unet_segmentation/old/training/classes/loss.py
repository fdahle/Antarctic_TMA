from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as func

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, un-normalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return Tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = func.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        print("ALL ROWS")
        print(all_rows)
        print("Y")
        print(y)
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred_ = func.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        #label_one_hot = label_one_hot.permute(0, 3, 1, 2)

        nce = -1 * torch.sum(label_one_hot * pred_, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()


class IouLoss(nn.Module):

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = 1e-6

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # If an ignore_index is provided, set the probabilities for that class to 0
        if self.ignore_index is not None:
            x = x.clone()  # clone to avoid changing the original tensor
            x[:, self.ignore_index, :, :] = 0

        # Apply argmax to get the most probable class for each pixel
        x_max = x.argmax(dim=1, keepdim=True)

        # One-hot encode the predictions to match the shape of y
        x_one_hot = torch.zeros_like(x)
        x_one_hot.scatter_(1, x_max, 1)

        # If an ignore_index is provided, ignore that class in the ground truth as well
        if self.ignore_index is not None:
            y = torch.where(y == self.ignore_index, 0, y)  # convert ignore_index to 0 in ground truth

        # One-hot encode the ground truth
        y_one_hot = torch.zeros_like(x)
        y_one_hot.scatter_(1, y.unsqueeze(1), 1)

        # Calculate IoU for each class
        intersection = (x_one_hot & y_one_hot).float().sum(dim=(2, 3))
        union = (x_one_hot | y_one_hot).float().sum(dim=(2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Ignore the ignored class during the IoU calculation
        if self.ignore_index is not None:
            iou = iou[:, :self.ignore_index]  # Exclude the ignored class from IoU calculation

        # Average the IoU across all classes (excluding the ignored class)
        loss = 1 - iou.mean(dim=1)  # Inverting the IoU to make it a loss

        return loss.mean()