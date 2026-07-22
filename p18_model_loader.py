
import torch
import torchvision
from torch.nn import Sequential, MaxPool2d, Conv2d

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
#加载方式1

model = torch.load("vgg16_method1.pth")
#print(model)

#加载方式2
model2 = torch.load("vgg16_method2.pth")
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.state_dict(torch.load("vgg16_method2.pth"))
#print(vgg16)


#小陷阱
model3 = torch.load("tudui_method.pth")
print(model3)