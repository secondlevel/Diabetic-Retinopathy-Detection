import pandas as pd
from skimage import io
import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


def process_data_number(img,label):
    img_0,img_1,img_2,img_3,img_4 = [],[],[],[],[]
    label_0,label_1,label_2,label_3,label_4 = [],[],[],[],[]

    print("before:",img.shape,label.shape)
    for i in range(5):
        for j in np.argwhere(label==i):
            if i == 0:
                img_0.append(img[j[0]])
                label_0.append(label[j[0]])

            elif i == 1:
                img_1.append(img[j[0]])
                label_1.append(label[j[0]])
            
            elif i == 2:
                img_2.append(img[j[0]])
                label_2.append(label[j[0]])
            
            elif i == 3:
                img_3.append(img[j[0]])
                label_3.append(label[j[0]])

            else:
                img_4.append(img[j[0]])
                label_4.append(label[j[0]])

    print(np.array(img_0).shape)
    print(np.array(label_0).shape)
    print(np.array(img_1).shape)
    print(np.array(label_1).shape)
    print(np.array(img_2).shape)
    print(np.array(label_2).shape)
    print(np.array(img_3).shape)
    print(np.array(label_3).shape)
    print(np.array(img_4).shape)
    print(np.array(label_4).shape)

    img_0 = img_0[:len(img_0)//5]
    label_0 = label_0[:len(label_0)//5]

    # img_1 = img_1[:len(img_4)]
    # label_1 = label_1[:len(label_4)]

    # img_2 = img_2[:len(img_4)]
    # label_2 = label_2[:len(label_4)]

    # img_3 = img_3[:len(img_4)]
    # label_3 = label_3[:len(label_4)]

    # print(np.array(img_0).shape)
    # print(np.array(label_0).shape)
    # print(np.array(img_1).shape)
    # print(np.array(label_1).shape)
    # print(np.array(img_2).shape)
    # print(np.array(label_2).shape)
    # print(np.array(img_3).shape)
    # print(np.array(label_3).shape)
    # print(np.array(img_4).shape)
    # print(np.array(label_4).shape)

    img,label=[],[]

    img = (img_0+img_1+img_2+img_3+img_4)
    label = (label_0+label_1+label_2+label_3+label_4)

    np.random.seed(0)
    np.random.shuffle(img)
    np.random.seed(0)
    np.random.shuffle(label)
    
    # print(np.array(img).shape)
    # print(np.array(label).shape)
    return np.array(img),np.array(label)

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv',header=None)
        label = pd.read_csv('train_label.csv',header=None)

        img,label = np.squeeze(img.values), np.squeeze(label.values)
        # img,label = process_data_number(img,label)
        return img,label
    else:
        img = pd.read_csv('test_img.csv',header=None)
        label = pd.read_csv('test_label.csv',header=None)
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode, transform=None):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.labels = getData(mode)
        self.transform = transform
        self.mode = mode
        print("> Found %d %s images..." % (len(self.img_name),self.mode))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        image_path = self.root + self.img_name[index]+ '.jpeg'
        self.img = io.imread(image_path)
        self.label = self.labels[index]

        if self.transform:
            self.img = self.transform(self.img)

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        return self.img,self.label

# if __name__ == "__main__":
#     img,label = getData("train")
