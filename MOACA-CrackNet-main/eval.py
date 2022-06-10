import torch
from torch.utils.data import DataLoader
import models.unetOctaveDscplus_7
import models.unet
from torch.autograd import Variable
import torchvision.transforms as T
import dataset

def eval():

    net = models.unet.UNet(1,1)
    net.eval()
    net.cuda()
    test_loss = 0
    correct = 0
    total = 0

    classnum = 2
    target_num = torch.zeros((1,classnum))
    predict_num = torch.zeros((1,classnum))
    acc_num = torch.zeros((1,classnum))
    criterion = torch.nn.BCELoss()
    criterion.cuda()

    loader = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5]), T.Resize((256, 256))])
    loader_1 = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5]), T.Resize((512, 512))])
    crack_dataset = dataset.CrackDataset('res', 'Unet400', transform=loader_1,
                                         target_transform=loader_1)

    testloader = DataLoader(crack_dataset, batch_size=2, shuffle=False, num_workers=4)

    use_cuda = 1

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)

    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)

#精度调整
    recall = (recall.numpy()[0]*100).round(3)
    precision = (precision.numpy()[0]*100).round(3)
    F1 = (F1.numpy()[0]*100).round(3)
    accuracy = (accuracy.numpy()[0]*100).round(3)

# 打印格式方便复制
    print('recall'," ".join('%s' % id for id in recall))
    print('precision'," ".join('%s' % id for id in precision))
    print('F1'," ".join('%s' % id for id in F1))
    print('accuracy',accuracy)

if __name__ == '__main__':
    eval()