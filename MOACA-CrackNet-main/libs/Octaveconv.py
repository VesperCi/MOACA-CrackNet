import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
 
up_kwargs = {'mode': 'nearest'}


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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class OctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.25, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = nn.Sequential(
            nn.Conv2d(int(alpha * in_ch), int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )
        self.l2h = nn.Sequential(
            nn.Conv2d(int(alpha * in_ch), out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch - int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )
        self.h2l = nn.Sequential(
            nn.Conv2d(in_ch - int(alpha * in_ch), int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )

        self.h2h = nn.Sequential(
            nn.Conv2d(in_ch - int(alpha * in_ch),
                                   out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch - int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        X_h, X_l = x
        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)
        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = self.upsample(X_l2h)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l
        return X_h, X_l

class FirstOctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.25, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = nn.Sequential(
            nn.Conv2d(in_ch, int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )
        self.h2h = nn.Sequential(
            nn.Conv2d(in_ch, out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch - int(alpha * out_ch)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.25, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = nn.Sequential(
            nn.Conv2d(int(alpha * in_ch), out_ch,
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.h2h = nn.Sequential(
            nn.Conv2d(in_ch - int(alpha * in_ch),
                                   out_ch,
                                   kernel_size, 1, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = self.l2h(X_l)
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(X_l2h)

        X_h = X_h2h + X_l2h
        return X_h