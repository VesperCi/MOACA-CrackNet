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
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_ch), int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_ch), out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_ch - int(alpha * in_ch), int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_ch - int(alpha * in_ch),
                                   out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        # if self.stride == 2:
        #     X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

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
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_ch, int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_ch, out_ch - int(alpha * out_ch),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        # if self.stride == 2:
        #     x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_ch), out_ch,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_ch - int(alpha * in_ch),
                                   out_ch,
                                   kernel_size, 1, padding, dilation, groups, bias)
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

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        #self.conv1 = DoubleConv(in_ch, 64)
        self.OCconvf1 = FirstOctaveConv(kernel_size=(3, 3), in_ch=in_ch, out_ch=64, bias=False, stride=2, alpha=0.75 ).cuda()
        self.OCconv1 = OctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvl1 = LastOctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvf2 = FirstOctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, stride=2, alpha=0.75 ).cuda()
        self.OCconv2 = OctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=128, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvl2 = LastOctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=128, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvf3= FirstOctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=256, bias=False, stride=2, alpha=0.75 ).cuda()
        self.OCconv3 = OctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=256, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvl3 = LastOctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=256, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvf4 = FirstOctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=512, bias=False, stride=2, alpha=0.75 ).cuda()
        self.OCconv4 = OctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=512, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvl4 = LastOctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=512, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvf5 = FirstOctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=1024, bias=False, stride=2, alpha=0.75 ).cuda()
        self.OCconv5 = OctaveConv(kernel_size=(3, 3), in_ch=1024, out_ch=1024, bias=False, stride=2, alpha=0.75).cuda()
        self.OCconvl5 = LastOctaveConv(kernel_size=(3, 3), in_ch=1024, out_ch=1024, bias=False, stride=2, alpha=0.75).cuda()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)
        # self.OCconvf1 = FirstOctaveConv(kernel_size=(3, 3), in_ch=in_ch, out_ch=64, bias=False, stride=2, alpha=0.75 ).cuda()
        # self.OCconv1 = OctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False, stride=2, alpha=0.75).cuda()
        # self.OCconvl1 = LastOctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False, stride=2, alpha=0.75).cuda()
        # self.OCconvf2 = Octconv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, stride=2,  ).cuda()
        # self.OCconv2 = Octconv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, stride=2, ).cuda()
        # self.OCconvl2 = Octconv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, stride=2, ).cuda()
        # self.OCconv3 = Octconv(kernel_size=(3, 3), in_ch=128, out_ch=256, bias=False, stride=2,  ).cuda()
        # self.OCconv4 = Octconv(kernel_size=(3, 3), in_ch=256, out_ch=512, bias=False, stride=2,  ).cuda()
        # self.OCconv5 = Octconv(kernel_size=(3, 3), in_ch=512, out_ch=1024, bias=False, stride=2,  ).cuda()
        # self.OCconv6 = Octconv(kernel_size=(3, 3), in_ch=1024, out_ch=512, bias=False, stride=2,  ).cuda()
        # self.OCconv7 = Octconv(kernel_size=(3, 3), in_ch=512, out_ch=256, bias=False, stride=2,  ).cuda()
        # self.OCconv8 = Octconv(kernel_size=(3, 3), in_ch=256, out_ch=128, bias=False, stride=2,  ).cuda()
        # self.OCconv9 = Octconv(kernel_size=(3, 3), in_ch=128, out_ch=64, bias=False, stride=2,  ).cuda()
        # self.OCconv10 = Octconv(kernel_size=(3, 3), in_ch=64, out_ch=out_ch, bias=False, stride=2,  ).cuda()



    def forward(self, x):
        c1_1 = self.OCconvf1(x)
        c1_2 = self.OCconv1(c1_1)
        c1 = self.OCconvl1(c1_2)
        p1 = self.pool1(c1)
        #p1 = self.pool1(c1)
        c2_1 = self.OCconvf2(c1)
        c2_2 = self.OCconv2(c2_1)
        c2 = self.OCconvl2(c2_2)
        p2 = self.pool2(c2)
        c3_1 = self.OCconvf3(c2)
        c3_2 = self.OCconv3(c3_1)
        c3 = self.OCconvl3(c3_2)
        p3 = self.pool3(c3)
        c4_1 = self.OCconvf4(c3)
        c4_2 = self.OCconv4(c4_1)
        c4 = self.OCconvl4(c4_2)
        p4 = self.pool4(c4)
        c5_1 = self.OCconvf5(c4)
        c5_2 = self.OCconv5(c5_1)
        c5 = self.OCconvl5(c5_2)

        #p2 = self.pool2(c2)
        # c3 = self.conv3(p2)
        # p3 = self.pool3(c3)
        # c4 = self.conv4(p3)
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)

        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)  # 化成(0~1)区间
        return out

if __name__ == '__main__':
    unet = UNet(1,1)
    print(unet)
    parameters = filter(lambda p: p.requires_grad, unet.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)