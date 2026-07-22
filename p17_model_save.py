import torch
import torchvision
from torch.nn import Sequential, Conv2d, MaxPool2d

vgg16 = torchvision.models.vgg16(pretrained=False)

#保存方式1 保存模型结构及参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2 保存参数
torch.save(vgg16.state_dict(),"vgg16_method2.pth")


#小陷阱
class tudui(torch.nn.Module):
    def __init__(self):
        super(tudui, self).__init__()
        self.module1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2)
        )

    def forward(self, input):
        input = self.module1(input)
        return input

t1 = tudui()
torch.save(t1,"tudui_method.pth")
