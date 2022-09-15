import os.path
import numpy.random as random
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.utils.data as data
class ImageSet(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageSet, self).__init__(root, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        img_path,class_id = self.imgs[index]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = self.transform(img)
        return img,class_id,img_name

    def __len__(self):
        return len(self.imgs)

class Pair_ImageSet(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageSet, self).__init__(root, transform=transform)
        self.transform = transform
        self.class_img_dict={}
        for img_path,class_id in self.imgs:
            if class_id not in self.class_img_dict.keys():
                self.class_img_dict[class_id]=[]
            self.class_img_dict[class_id].append(img_path)

    def __getitem__(self, index):
        img_path,class_id = self.imgs[index]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = self.transform(img)

        same_class_img_list = self.class_img_dict[class_id]
        pos_img_path = same_class_img_list[random.randint(len(same_class_img_list))]
        pos_img = Image.open(pos_img_path)
        pos_img = self.transform(pos_img)
        posimg_name = pos_img_path.split('/')[-1]
        return img,pos_img,class_id,img_name,posimg_name

    def __len__(self):
        return len(self.imgs)

class TB_ImageSet(data.Dataset):
    def __init__(self, root,file_path, transform=None):
        
        self.transform = transform
        self.img_list = []
        self.label_list = []
        if os.path.exists(file_path):
            with open(file_path,'r') as fp:
                data = fp.readlines()
                for line in data:
                    line_cell = line.strip().split(' ')
                    if len(line_cell)>2:
                        img_name = " ".join(line_cell[:-1])
                        label = int(line_cell[-1])
                    else:
                        img_name = line_cell[0]
                        label = int(line_cell[1])
                    img_path = os.path.join(root,img_name)
                    self.img_list.append(img_path)
                    self.label_list.append(label)
        else:
            print("could not find file %s"%file_path)


    def __getitem__(self, index):
        img_path = self.img_list[index]
        class_id = self.label_list[index]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = self.transform(img)
        
        return img,class_id,img_name

    def __len__(self):
        return len(self.img_list)


class Pair_TB_ImageSet(data.Dataset):
    def __init__(self, root, file_path, transform=None):

        self.transform = transform
        self.img_list = []
        self.label_list = []
        self.class_img_dict = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as fp:
                data = fp.readlines()
                for line in data:
                    line_cell = line.strip().split(' ')
                    if len(line_cell) > 2:
                        img_name = " ".join(line_cell[:-1])
                        label = int(line_cell[-1])
                    else:
                        img_name = line_cell[0]
                        label = int(line_cell[1])
                    img_path = os.path.join(root, img_name)
                    self.img_list.append(img_path)
                    self.label_list.append(label)

                    if label not in self.class_img_dict.keys():
                        self.class_img_dict[label]=[]
                    self.class_img_dict[label].append(img_path)
        else:
            print("could not find file %s" % file_path)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        class_id = self.label_list[index]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = self.transform(img)

        same_class_img_list = self.class_img_dict[class_id]
        pos_img_path = same_class_img_list[random.randint(len(same_class_img_list))]
        pos_img = Image.open(pos_img_path)
        pos_img = self.transform(pos_img)
        posimg_name = pos_img_path.split('/')[-1]
        return img,pos_img, class_id, img_name,posimg_name

    def __len__(self):
        return len(self.img_list)