import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(
    nn.Linear(20,256)
    ,nn.ReLU()
    ,nn.Linear(256,10)
)

X = torch.rand(2,20)
print(net(X))
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
print(net(X))

class MySequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x

net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(torch.mm(x,self.rand_weight) + 1)
        x = self.linear(x)
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
net = FixedHiddenMLP()
print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.linear = nn.Linear(32,16)

    def forward(self, x):
        return self.linear(self.net(x))

chim = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
chim(X)