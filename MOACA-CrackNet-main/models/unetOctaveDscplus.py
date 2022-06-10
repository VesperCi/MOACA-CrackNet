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

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        #self.conv1 = DoubleConv(in_ch, 64)
        self.OCconvf1 = FirstOctaveConv(kernel_size=(3, 3), in_ch=in_ch, out_ch=64, bias=False,alpha=0.25).cuda()
        self.OCconv1 = OctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False,alpha=0.25).cuda()
        self.OCconvl1 = LastOctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=64, bias=False, stride=2, alpha=0.25).cuda()
        #self.OCconvf2 = FirstOctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, stride=2, alpha=0.25 ).cuda()
        self.OCconv2 = OctaveConv(kernel_size=(3, 3), in_ch=64, out_ch=128, bias=False, alpha=0.25).cuda()
        self.OCconvl2 = LastOctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=128, bias=False, stride=2, alpha=0.25).cuda()
        #self.OCconvf3= FirstOctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=256, bias=False, stride=2, alpha=0.25 ).cuda()
        self.OCconv3 = OctaveConv(kernel_size=(3, 3), in_ch=128, out_ch=256, bias=False, alpha=0.25).cuda()
        self.OCconvl3 = LastOctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=256, bias=False, stride=2, alpha=0.25).cuda()
        #self.OCconvf4 = FirstOctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=512, bias=False, stride=2, alpha=0.25 ).cuda()
        self.OCconv4 = OctaveConv(kernel_size=(3, 3), in_ch=256, out_ch=512, bias=False, alpha=0.25).cuda()
        self.OCconvl4 = LastOctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=512, bias=False, stride=2, alpha=0.25).cuda()
        #self.OCconvf5 = FirstOctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=1024, bias=False, stride=2, alpha=0.25 ).cuda()
        self.OCconv5 = OctaveConv(kernel_size=(3, 3), in_ch=512, out_ch=1024, bias=False, alpha=0.25).cuda()
        self.OCconvl5 = LastOctaveConv(kernel_size=(3, 3), in_ch=1024, out_ch=1024, bias=False, stride=2, alpha=0.25).cuda()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv2 = DoubleConv(64, 128)
        #self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        #self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        #self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = Dsconv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = Dsconv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = Dsconv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = Dsconv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.CA = CoordAtt(1024 , 1024)
        self.CA4 = CoordAtt(512, 512)
        self.CA3 = CoordAtt(256, 256)
        self.CA2 = CoordAtt(128, 128)
        self.CA1 = CoordAtt(64, 64)


    def forward(self, x):
        c1_1 = self.OCconvf1(x)
        c1_2 , c1_3 = self.OCconv1(c1_1)
        c1_4 = c1_2 , c1_3
        p1_1 = self.pool(c1_2)
        p1_2 = self.pool(c1_3)
        c1_5 = p1_1 , p1_2
        c1_6 = self.OCconvl1(c1_4)
        c1 = self.CA1(c1_6)
        #p1 = self.pool1(c1)
        #c2_1 = self.OCconvf2(c1)
        c2_2 ,c2_3 = self.OCconv2(c1_5)
        c2_4 = c2_2, c2_3
        p2_1 = self.pool(c2_2)
        p2_2 = self.pool(c2_3)
        c2_5 = p2_1 , p2_2
        c2_6 = self.OCconvl2(c2_4)
        c2 = self.CA2(c2_6)
       # p2 = self.pool2(c2)
        #c3_1 = self.OCconvf3(c2)
        c3_2 ,c3_3 = self.OCconv3(c2_5)
        c3_4 = c3_2, c3_3
        p3_1 = self.pool(c3_2)
        p3_2 = self.pool(c3_3)
        c3_5 = p3_1 , p3_2
        c3_6 = self.OCconvl3(c3_4)
        c3 = self.CA3(c3_6)
        #p3 = self.pool3(c3)
        #c4_1 = self.OCconvf4(c3)
        c4_2 , c4_3 = self.OCconv4(c3_5)
        c4_4 = c4_2, c4_3
        p4_1 = self.pool(c4_2)
        p4_2 = self.pool(c4_3)
        c4_5 = p4_1 , p4_2
        c4_6 = self.OCconvl4(c4_4)
        c4 = self.CA4(c4_6)
        #p4 = self.pool4(c4)
       # c5_1 = self.OCconvf5(c4)
        c5_2 , c5_3 = self.OCconv5(c4_5)
        c5_4 = c5_2, c5_3
        # p5_1 = self.pool(c5_2)
        # p5_2 = self.pool(c5_3)
        # c5_5 = p5_1 , p5_2
        c5_1 = self.OCconvl5(c5_4)
        c5 = self.CA(c5_1)

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