from curses import def_shell_mode
from dataclasses import dataclass
from fileinput import filename
from multiprocessing import BoundedSemaphore
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path as osp
from PIL import Image
import pickle
from glob import glob
from itertools import islice
import veri_train, Rotation

def directory_paths():
    V = veri_train.VeRi()
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    train_dir = osp.join(dataset_dir, 'image_train')
    train_list = None   # I think data gets shuffled using this  # 37778
    test_dir = osp.join(dataset_dir,'image_test')
    test_list = None
    Dsl_path = osp.join(dataset_dir,'Dsl/')
    Dsl_test_path = osp.join(dataset_dir, 'Dsl_test/')
    root_dir = osp.join(dataset_dir,'Dsl')
    return V, dataset_dir, train_dir, train_list, test_dir, train_list, test_list, Dsl_path, Dsl_test_path, root_dir

V, dataset_dir, train_dir, train_list, test_dir, train_list, test_list, Dsl_path, Dsl_test_path, root_dir = directory_paths()

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

'''def Data_Rotation(Train_Images,Data_Size):
    Dsl = []; Dsl_Label = []
    for i in range(Data_Size):
        # image = mpimg.imread(Train_Images[i])
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        R_0 = Rotation._apply_2d_rotation(_4d_tensor,0)
        R_90 = Rotation._apply_2d_rotation(_4d_tensor,90)
        R_180 = Rotation._apply_2d_rotation(_4d_tensor,180)
        R_270 = Rotation._apply_2d_rotation(_4d_tensor,270)
        Dsl.append(R_0)
        Dsl_Label.append(0)
        Dsl.append(R_90)
        Dsl_Label.append(1)
        Dsl.append(R_180)
        Dsl_Label.append(2)
        Dsl.append(R_270)
        Dsl_Label.append(3)
    Dsl_Label = torch.Tensor(Dsl_Label)
    return Dsl, Dsl_Label'''

def Data_List(Train_Images,Data_Size):  #for new train
    Dsl = []; Dsl_Label = []
    for i in range(Data_Size):
        # image = mpimg.imread(Train_Images[i])
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        Dsl.append(_4d_tensor)
    return Dsl

'''def get_data(No_of_Train_Images, No_of_Test_Images):
    Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
    Dsl, Dsl_Label= Data_Rotation(Train_Images,No_of_Train_Images)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(test_dir,test_list)
    Dsl_test, Dsl_Label_test = Data_Rotation(Test_Images,No_of_Test_Images)
    return Dsl, Dsl_Label, Dsl_test, Dsl_Label_test'''

def get_data(No_of_Train_Images, No_of_Test_Images):
    Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
    Dsl = Data_List(Train_Images,No_of_Train_Images)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(test_dir,test_list)
    Dsl_test = Data_List(Test_Images,No_of_Test_Images)
    return Dsl, Dsl_test

Dsl, Dsl_test = get_data(28,28)  #4000,1120

def save_pkl(D,path):
    with open(path, 'wb') as f:
        pickle.dump(D, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl_folder(Data, path):
    for i in range(len(Data)):
        D = {}
        D['image'] = Data[i]
        # D['label'] = Data_Label[i]
        tmp = path + f'{i}.pkl'
        save_pkl(D,tmp)

save_pkl_folder(Dsl,Dsl_path)
save_pkl_folder(Dsl_test, Dsl_test_path)

class Veri(Dataset):
    """dataset."""
    def __init__(self, root_dir, transform=None):
        self.files = glob(f'{root_dir}*.pkl')
        self.transform = transform
        self.class_names = ['0','1','2','3']  # 0 for '0', 1 for '90', 2 for '180', 3 for '270'

    def __len__(self):
        return len(self.files)  # partial data, return = 10

    def __getitem__(self, idx):
        file = self.files[idx]
        D = read_pkl(file)
        return {
            'image': torch.tensor(D['image']),
            'label': torch.tensor(D['label'], dtype=torch.long),
            'class_names': self.class_names
        }

def data_loader(path,batch_size):
    veri = Veri(path)
    loader = torch.utils.data.DataLoader(veri, batch_size, shuffle=True)
    return loader,veri 