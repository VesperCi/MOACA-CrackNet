import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Dsconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dsconv, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out