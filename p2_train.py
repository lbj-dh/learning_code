
from torch.utils.data import Dataset
from PIL import Image
import os


class Mydataset(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)


root_dir = "C:\\Users\\邓宏\\Downloads\\hymenoptera_data\\hymenoptera_data\\train"
label_dir = "ants"

root1_dir = "C:\\Users\\邓宏\\Downloads\\hymenoptera_data\\hymenoptera_data\\train"
label1_dir = "bees"
ants_dataset = Mydataset(root_dir,label_dir)
bees_dataset = Mydataset(root1_dir,label1_dir)

train_dataset = ants_dataset + bees_dataset


