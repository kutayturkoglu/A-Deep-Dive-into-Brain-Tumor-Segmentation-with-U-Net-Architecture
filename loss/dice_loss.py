import torch.nn.functional as F
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, input, target):
        input = torch.sigmoid(input)

        intersection = torch.sum(input * target)
        union = torch.sum(input) + torch.sum(target)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice_coeff

        return dice_loss

