import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class tudui(torch.nn.Module):
    def __init__(self):
        super(tudui, self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)
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
        output = self.conv1(input)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        output = self.flatten(output)
        output = self.linear1(output)
        output = self.linear2(output)
        input = self.module1(input)
        return input


tudui = tudui()
print(tudui)
input = torch.ones(64,3,32,32)
output = tudui(input)
print(output.size())

writer = SummaryWriter("./logs")
writer.add_graph(tudui,input)
writer.close()