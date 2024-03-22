import torch.nn.functional as F
from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        inputs = torch.sigmoid(input)
        smooth = 1.0
        inputs = inputs.reshape(-1)
        tar = target.reshape(-1)
        intersection = (inputs*tar).sum()

        dic_loss = 1-((2.0*intersection+smooth)/(inputs.sum()+tar.sum()+smooth))

        return dic_loss
