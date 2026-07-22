import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader

import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


# 数据集
# 训练数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
# 测试数据集
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# 数据集长度
train_len = len(train_dataset)
test_len = len(test_dataset)

print(train_len)
print(test_len)

# 加载数据集
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 搭建神经网络
class Tudui(torch.nn.Module):
    def __init__(self):
        super(tudui, self).__init__()
        self.module1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self, input):
        input = self.module1(input)
        return input
tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数
lose_fun = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    lose_fun = lose_fun.cuda()

# 优化器
learning_rate = 0.001
optimizer = optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络中的参数
# 记录训练的次数
train_step = 0
# 记录测试的次数
test_step = 0

# 添加tensorboard
writer = SummaryWriter("./logs_test")

# 训练的轮数
epoch = 10
start_time = time.time()
for i in range(epoch):
    # 训练步骤开始
    print("-----第{}轮训练开始-----".format(i+1))

    tudui.train()
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        ouput = tudui(imgs)
        loss = lose_fun(ouput, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{}，loss:{}".format(train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(),train_step)

    # 测试步骤开始
    # 测试总损失
    total_loss_sum = 0
    total_accuracy = 0
    tudui.eval()
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            ouput = tudui(imgs)
            loss = lose_fun(ouput, targets)
            total_loss_sum = total_loss_sum + loss.item()
            accuracy = (ouput.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的总损失为：{}".format(total_loss_sum))
    print("整体测试集上的正确率为：{}".format(total_accuracy/test_len))
    test_step += 1
    writer.add_scalar("test_loss",total_loss_sum,test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_len,test_step)

    torch.save(tudui,"tudui{}.pth".format(i+1))
    print("模型已保存")
writer.close()