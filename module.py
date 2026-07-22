import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class tudui(torch.nn.Module):
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