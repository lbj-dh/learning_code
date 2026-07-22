import torchvision
from torch.utils.tensorboard import SummaryWriter

dateset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,transform=dateset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,transform=dateset_transform, download=False)

writer = SummaryWriter("dataset")
for i in range (10):
    img, target = train_set[i]
    writer.add_image("test_set",img,i)

writer.close()
