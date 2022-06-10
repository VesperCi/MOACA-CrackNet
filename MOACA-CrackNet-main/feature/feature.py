import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

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
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride == 2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels,
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


transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
train_set = tv.datasets.ImageFolder(root='./', transform=transform)
data_loader = DataLoader(dataset=train_set)

to_pil_image = transforms.ToPILImage()

for image, label in data_loader:
    # 方法1：Image.show()
    # transforms.ToPILImage()中有一句
    # npimg = np.transpose(pic.numpy(), (1, 2, 0))
    # 因此pic只能是3-D Tensor，所以要用image[0]消去batch那一维
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_channels=3, out_channels=64,stride=2,alpha=0.125)
    OCconv = OctaveConv(kernel_size=(3, 3), in_channels=64, out_channels=64, stride=2, alpha=0.125)
    LOCconv = LastOctaveConv(kernel_size=(3, 3), in_channels=64, out_channels=3, stride=2, alpha=0.125)
    x_out, y_out = FOCconv(image)

    #x = to_pil_image(x_out)
    img = to_pil_image(image[0])
    conv1 = nn.Conv2d(56,1,1)
    conv2 = nn.Conv2d(8,1,1)
    c_1 = x_out, y_out
    c_2 , c_3 = OCconv(c_1)

    x1 = conv1(x_out)
    y1 = conv2(y_out)
    x2 = conv1(c_2)
    y2 = conv2(c_3)
    x_1 = to_pil_image(x2[0])
    y_1 = to_pil_image(y2[0])
    x = to_pil_image(x1[0])
    y = to_pil_image(y1[0])
   # print("First: ", x_out.size(), y_out.size())
   # print("First: ", image.size())
    x.show()
    y.show()
    x_1.show()
    y_1.show()

    # 方法2：plt.imshow(ndarray)
    # img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    # img = img.numpy()  # FloatTensor转为ndarray
    # img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后

    # 显示图片
    plt.imshow(img)

'''
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
    FOCconv = FirstOctaveConv(kernel_size=(3, 3), in_channels=3, out_channels=64,stride=2,alpha=0.25).cuda()
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
'''