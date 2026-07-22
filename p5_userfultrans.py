from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_url = Image.open("image/36900412_92b81831ad.jpg")

print(img_url)

to_tensor = transforms.ToTensor()
print(to_tensor)

img_tensor= to_tensor(img_url)

print(img_tensor)

writer.add_image("test", img_tensor)


tn = transforms.Normalize([0.2,0.5,0.9],[0.7,0.8,0.9])

tn1 = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = tn1(img_tensor)
writer.add_image("change", img_norm,3)


tr = transforms.Resize((512,512))
img_resize = tr(img_url)
img_resize = to_tensor(img_resize)
writer.add_image("resize", img_resize,3)

tr_2 = transforms.Resize(512)
img_new = transforms.Compose([tr_2,to_tensor])
tr_2 = img_new(img_url)
writer.add_image("resize222", tr_2,3)


tr_3 = transforms.RandomCrop(50)
tr_3 = transforms.Compose([tr_3,to_tensor])

for i in range (10):
    img_crop = tr_3(img_url)
    writer.add_image("crop", img_crop,i)







writer.close()

