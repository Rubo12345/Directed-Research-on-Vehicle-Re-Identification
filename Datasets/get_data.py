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
import sys
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/Datasets/')
import veri_train
import xml.etree.ElementTree as ET
import classes

def directory_paths():
    V = veri_train.VeRi()
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    train_dir = osp.join(dataset_dir, 'image_test')
    train_list = osp.join(dataset_dir, 'name_test.txt') # IMP Change
    test_dir = osp.join(dataset_dir,'image_query')
    test_list = osp.join(dataset_dir, 'name_query.txt')  # IMP Change
    Dsl_path = osp.join(dataset_dir,'Dsl/')
    Dsl_test_path = osp.join(dataset_dir, 'Dsl_test/')
    root_dir = osp.join(dataset_dir,'Dsl')
    return V, dataset_dir, train_dir, train_list, test_dir, test_list,Dsl_path, Dsl_test_path,root_dir

V, dataset_dir, train_dir, train_list, test_dir,test_list,Dsl_path, Dsl_test_path,root_dir = directory_paths()

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

class_labels = [0, 2, 3, 5, 7, 8, 10, 14, 15, 16, 21, 22, 24, 25, 30, 36, 38, 39, 42, 46, 47, 49, 50, 51, 53, 54, 57, 60, 61, 64, 66, 67, 68, 70, 71, 72, 76, 77, 78, 79, 84, 85, 88, 91, 94, 95, 96, 97, 100, 101, 102, 103, 104, 108, 109, 110, 111, 114, 116, 120, 122, 123, 125, 126, 127, 128, 130, 131, 138, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 1, 4, 6, 9, 11, 12, 13, 17, 18, 19, 20, 23, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 40, 41, 43, 44, 45, 48, 52, 55, 56, 58, 59, 62, 63, 65, 69, 73, 74, 75, 80, 81, 82, 83, 86, 87, 89, 90, 92, 93, 98, 99, 105, 106, 107, 112, 113, 115, 117, 118, 119, 121, 124, 129, 132, 133, 134, 135, 136, 137, 139, 140, 141]
class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]

def Data_List_Train(Train_Images,Train_Labels,Data_Size):  #for new train
    Dsl= []; Dsl_Label = []
    for i in range(Data_Size):
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        Dsl.append(_4d_tensor)
        label = class_index[class_labels.index(Train_Labels[i])]
        Dsl_Label.append(label)
    Dsl_Label = torch.Tensor(Dsl_Label)
    return Dsl,Dsl_Label

def Data_List_Test(Test_Images,Test_Labels,Data_Size): 
    Dsl_test = []; Dsl_Label_test = []
    for i in range(Data_Size):
        _4d_tensor = input_to_4d_tensor(Test_Images[i])
        Dsl_test.append(_4d_tensor)
        label = class_index[class_labels.index(Test_Labels[i])]
        Dsl_Label_test.append(label)
    Dsl_Label_test = torch.Tensor(Dsl_Label_test)
    return Dsl_test,Dsl_Label_test

def get_the_data(No_of_Train_Images, No_of_Test_Images):
    Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
    Dsl,Dsl_Label = Data_List_Train(Train_Images,Train_Labels,No_of_Train_Images)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(test_dir,test_list)
    Dsl_test,Dsl_Label_test = Data_List_Test(Test_Images,Test_Labels,No_of_Test_Images)
    return Dsl, Dsl_Label, Dsl_test, Dsl_Label_test

Dsl, Dsl_Label, Dsl_test, Dsl_Label_test = get_the_data(11576,1676) # 11579,1678

def save_pkl(D,path):
    with open(path, 'wb') as f:
        pickle.dump(D, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl_folder(Data, Data_Label, path):
    for i in range(len(Data)):
        D = {}
        D['image'] = Data[i]
        D['label'] = Data_Label[i]
        tmp = path + f'{i}.pkl'
        save_pkl(D,tmp)

save_pkl_folder(Dsl,Dsl_Label,Dsl_path)
save_pkl_folder(Dsl_test,Dsl_Label_test, Dsl_test_path)

class Veri(Dataset):
    """dataset."""
    def __init__(self, root_dir, transform=None):
        self.files = glob(f'{root_dir}*.pkl')
        self.transform = transform
        
    def __len__(self):
        return len(self.files) 

    def __getitem__(self, idx):
        file = self.files[idx]
        D = read_pkl(file)
        return {
            'image': D['image'].clone().detach(),
            'label': D['label'].clone().detach().type(torch.LongTensor)
        }

def data_loader(path,batch_size,b):
    veri = Veri(path)
    loader = torch.utils.data.DataLoader(veri, batch_size, shuffle=b)
    return loader,veri 

