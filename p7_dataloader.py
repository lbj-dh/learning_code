import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from p11_torchvision_transform import target

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor())


test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0,drop_last=False)
img,target = test_set[0]
print(img)
print(target)


writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("epoch{}".format(epoch), imgs, step)
        step += 1

writer.close()


