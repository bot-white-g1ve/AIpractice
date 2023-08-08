import torchvision
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from time import time

from vgg16 import Tudui
from torch.utils.data import DataLoader

print(torch.__version__)

print("cuda available?:"+str(torch.cuda.is_available()))

#读取数据
train_data = torchvision.datasets.CIFAR10("CIFAR10Data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("CIFAR10Data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

#查看数据大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print("train_data_size is "+str(train_data_size))
print("test_data_size is "+str(test_data_size))

#加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#创建模型
tudui = Tudui().cuda()

#创建损失函数
loss_fn = nn.CrossEntropyLoss().cuda()

#创建优化器
optimizer = torch.optim.SGD(tudui.parameters(),lr=0.01)

#创建TensorBoard
writer = SummaryWriter("logs")

#记录训练步骤
train_step = 0
test_step = 0
epoch = 10

start_time = time()

for i in range(epoch):
    print("-----第"+str(i+1)+"轮训练开始-----")
    # 每一轮的训练
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step+=1
        if train_step%100 == 1:
            print("训练次数：" + str(train_step) + "，误差：" + str(loss.item()))
            writer.add_scalar("train_loss", loss.item(), train_step)

    #每一轮的测试
    tudui.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad(): #代表下面的所有流程不带梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    test_step += 1
    print("=====这一轮的总测试误差为"+str(total_loss)+"=====")
    writer.add_scalar("test_loss", total_loss, test_step)
    print("=====这一轮正确了"+str(total_accuracy)+",正确率为"+str(total_accuracy/test_data_size)+"=====")
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, test_step)

end_time = time()
print(end_time-start_time)

torch.save(tudui, "tudui.pth")