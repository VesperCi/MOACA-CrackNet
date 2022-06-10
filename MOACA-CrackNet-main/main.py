import torch
from torchvision.transforms import transforms as T
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from torch import optim
from dataset import CrackDataset
from torch.utils.data import DataLoader
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import  os
import models.Unet_CA
import models.unetOctave
import models.unet



import  cv2
import numpy as np
from torchvision import  utils as utils
import matplotlib as mp

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5], [0.5]),  # torchvision.transforms.Normalize(mean, std, inplace=False)
    T.Resize((512, 512))
])
# mask只需要转换为tensor
y_transform = T.Compose([
    T.ToTensor(),
    T.Resize((512 , 512))
])

epoch = 101


def train_model(model, criterion, optimizer, dataload, num_epochs=epoch):
    Loss = []
    itertion = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数

        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4

            itertion += 1
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            inputs.cuda()
            labels.cuda()
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        Loss.append(epoch_loss)

        if epoch==num_epochs-1:
            torch.save(model.state_dict(), 'weights_%d.pth' % epoch)  # 返回模型的所有内容

    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(Loss, label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
   # plt.show()
    return model
#Using a target size (torch.Size([2, 3, 512, 512])) that is different to the input size (torch.Size([2, 1, 512, 512]))

# 训练模型
def train():
    model = models.unet.UNet(1,1)
    model.cuda()
    batch_size = args.batch_size
    # 损失函数
    criterion = torch.nn.BCELoss()
    # 梯度下降
    optimizer = optim.Adam(model.parameters())  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    crack_dataset = CrackDataset(r'..', transform=x_transform,
                                 target_transform=y_transform)
    dataloader = DataLoader(crack_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # batch_size：how many samples per minibatch to load，这里为4，数据集大小400，所以一共有100个minibatch
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度 
    train_model(model, criterion, optimizer, dataloader)

# 测试
def test():
    loader = T.Compose([T.ToTensor(),T.Normalize([0.5], [0.5]), T.Resize((512, 512))])
    unloader = T.ToPILImage()

    model = models.unetODP.UNet(1,1)
    model.cuda()
    #model = CBAM_Unet.UNet(1,1)
    model.load_state_dict(torch.load('weights/unetODP.pth', map_location='cpu'))
    model.eval()
    #model.summary()
    os.makedirs('unetODP')

    for str in os.listdir('res'):
        path = 'res/'+str
        #print(path)
        image = Image.open(path).convert('L')
        image = loader(image).unsqueeze(0)
        image = image.to(torch.float).to(device)

        y = model(image)
        img_y = y.squeeze(0)
        img_y = unloader(img_y)
        fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        # 设置子图占满整个画布
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # 关掉x和y轴的显示
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_y, cmap='gray')
        '''
        注意，这里加上plt.show()后，保存的图片就为空白了，因为plt.show()之后就会关掉画布，
        所以如果要保存加显示图片的话一定要将plt.show()放在pltd.savefig(save_path)之后
        '''
        # plt.show()
        plt.savefig('unetODP/'+str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()
    #train()
    test()



