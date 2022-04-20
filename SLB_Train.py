from curses import def_shell_mode
from fileinput import filename
from multiprocessing import BoundedSemaphore
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path as osp
from PIL import Image
import sys
sys.path.append('/home/rutu/WPI/Directed_Research/My_Approach_To_DR') 
from Datasets import veri_train, Rotation
from torch.utils.data import Dataset, DataLoader
import pickle

V = veri_train.VeRi()
dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
train_dir = osp.join(dataset_dir, 'image_train')
train_list = None   # I think data gets shuffled using this  # 37778
# train_list = osp.join(dataset_dir, 'name_train.txt')       # 37746

def data_image_labels(train_dir, train_list):
        train_data = V.process_dir(train_dir,train_list, relabel=True)
        Train_Images = [];Train_Labels = [];Train_Cams = []
        for image in range(len(train_data)):
            Train_Images.append(train_data[image][0])
            Train_Labels.append(train_data[image][1])
            Train_Cams.append(train_data[image][2])
        return Train_Images, Train_Labels, Train_Cams

def image_to_pixel(image):
    pix_val = list(image.getdata())
    pix_val_flat = [x for sets in pix_val for x in sets]
    return pix_val_flat

def input_to_4d_tensor(I):
    ''' Function converts the image into a tensor as well as size it'''
    preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(I)
    Tensor = preprocess(input_image)
    Tensor = torch.reshape(Tensor,(1,3,Tensor.shape[1],Tensor.shape[2]))
    return Tensor

def Data_Rotation(Train_Images):
    Dsl = []; Dsl_Label = []
    for i in range(56):
        # image = mpimg.imread(Train_Images[i])
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        R_0 = Rotation._apply_2d_rotation(_4d_tensor,0)
        R_90 = Rotation._apply_2d_rotation(_4d_tensor,90)
        R_180 = Rotation._apply_2d_rotation(_4d_tensor,180)
        R_270 = Rotation._apply_2d_rotation(_4d_tensor,270)
        Dsl.append(R_0)
        Dsl_Label.append(0)
        Dsl.append(R_90)
        Dsl_Label.append(90)
        Dsl.append(R_180)
        Dsl_Label.append(180)
        Dsl.append(R_270)
        Dsl_Label.append(270)
    Dsl_Label = torch.Tensor(Dsl_Label)
    return Dsl, Dsl_Label

Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
Dsl, Dsl_Label= Data_Rotation(Train_Images)
Dsl_Label_28 = torch.reshape(Dsl_Label,(28,2*4))


def save_pkl(D,path):
    with open(path, 'wb') as f:
        pickle.dump(D, f)
    
def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

path = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl/'
for i in range(len(Dsl)):
    D = {}
    D['image'] = Dsl[i]
    D['label'] = Dsl_Label[i]
    tmp = path + f'{i}.pkl'
    save_pkl(D,tmp)

root_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl/'
from glob import glob
class Veri(Dataset):
    """dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = glob(f'{root_dir}*.pkl')
        self.transform = transform

    def __len__(self):
        return len(self.files)  # partial data, return = 10

    def __getitem__(self, idx):
        file = self.files[idx]
        D = read_pkl(file)
        return {
            'image': torch.tensor(D['image']),
            'label': torch.tensor(D['label'])
        }

veri = Veri('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl/')
veri_loader = torch.utils.data.DataLoader(veri, batch_size=28, shuffle=True)
print(len(veri_loader))

for index, dic in enumerate(veri_loader):
    print(index)
    print(dic['image'].size())
    print(dic['label'].size())
    print(dic['image'].squeeze().size())