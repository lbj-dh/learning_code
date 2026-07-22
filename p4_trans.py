from PIL import Image
import cv2

from torchvision import transforms

from torch.utils.tensorboard import  SummaryWriter

img_url = "C:\\Users\\邓宏\\Downloads\\hymenoptera_data\\hymenoptera_data\\train\\bees\\36900412_92b81831ad.jpg"

img = Image.open(img_url)

print(img)

totensor = transforms.ToTensor()
img_ten = totensor(img)



print(img_ten)

writer = SummaryWriter("logs")

writer.add_image("test", img_ten)

writer.close()
