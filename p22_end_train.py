import torch
import torchvision
from PIL import Image
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torchvision import transforms

img_url = "./image/airplane.png"

image = Image.open(img_url).convert("RGB")
print(image)

transform = transforms.Compose([torchvision.transforms.Resize((32,32))
                                , torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
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

model = torch.load('tudui_8.pth')

image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)

print(output)

print(output.argmax(1))