from torch.utils.tensorboard import  SummaryWriter

from PIL import Image

import numpy as np

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

img_url = "C:\\Users\\邓宏\\Downloads\\hymenoptera_data\\hymenoptera_data\\train\\bees\\36900412_92b81831ad.jpg"

img = Image.open(img_url)
img_arr = np.array(img)
writer.add_image("train",img_arr,2,dataformats='HWC')

writer.close()