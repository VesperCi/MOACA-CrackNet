import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

up_kwargs = {'mode': 'nearest'}

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
    def __init__(self, in_ch, out_ch, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(OctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_ch, in_ch, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.bias = torch.zeros(out_ch).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.bn_h = nn.BatchNorm2d(int(out_ch*(1-alpha_out)))
        self.bn_l = nn.BatchNorm2d(int(out_ch*alpha_out))
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)
        end_h_x = int(self.in_ch*(1- self.alpha_in))
        end_h_y = int(self.out_ch*(1- self.alpha_out))
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.interpolate(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        x_h = self.relu(self.bn_h(X_h))
        x_l = self.relu(self.bn_l(X_l))

        return x_h, x_l

class FirstOctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size, alpha_in=0.0, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(FirstOctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_ch, in_ch, kernel_size[0], kernel_size[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ch))
        else:
            self.bias = torch.zeros(out_ch).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.bn_h = nn.BatchNorm2d(int(out_ch * (1 - alpha_out)))
        self.bn_l = nn.BatchNorm2d(int(out_ch * alpha_out))
        self.relu = nn.ReLU(inplace=True)


if __name__ == '__main__':
    # # nn.Conv2d
    # high = torch.Tensor(1, 64, 32, 32).cuda()
    # low = torch.Tensor(1, 192, 16, 16).cuda()
    # # test Oc conv
    # OCconv = OctaveConv(kernel_size=(3,3),in_channels=256,out_channels=512,bias=False,stride=2,alpha=0.75).cuda()
    # i = high,low
    # print("大小: ", high.size(), low.size())
    # x_out,y_out = OCconv(i)
    # print(x_out.size())
    # print(y_out.size())
    i = torch.Tensor(1, 3, 512, 512).cuda()
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_ch=3, out_ch=64,stride=2,alpha_in=0.0, alpha_out=0.5).cuda()
    x_out, y_out = FOCconv(i)
    print("First: ", x_out.size(), y_out.size())
    # # test last Octave Cov
    # LOCconv = LastOctaveConv(kernel_size=(3, 3), in_channels=256, out_channels=128, alpha=0.75).cuda()
    # i = high, low
    # out = LOCconv(i)
    # print("Last: ", out.size())
    parameters = filter(lambda p: p.requires_grad, FOCconv.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)