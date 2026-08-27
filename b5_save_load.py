import torch
from torch import nn
from torch.nn import functional as F


x = torch.arange(4)
torch.save(x,'x-save')
x2 = torch.load('x-save')
print(x2)

y = torch.zeros(4)
torch.save([x,y],'dict_save')
x3,y3 = torch.load('dict_save')
print(x3)
print(y3)

mydict = {'x' : x, 'y' : y}
torch.save(mydict,'dt_save')
mydict2 = torch.load('dt_save')
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)

torch.save(net.state_dict(),'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y_clone == Y)

