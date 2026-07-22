import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops.misc import Conv2d

dataset = torchvision.datasets.CIFAR10(root="./data", train=False,transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset,batch_size=64)


class dh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cov = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,x):
        x = self.cov(x)
        return x


Dh = dh()

step = 0
writer = SummaryWriter("./logs")
for data in dataloader:
    img,target = data
    output = Dh.forward(img)
    writer.add_images("input",img,step)
    writer.add_images("output",output,step)
    step += 1

writer.close()


