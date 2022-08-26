from curses import def_shell_mode
from dataclasses import dataclass
from fileinput import filename
from multiprocessing import BoundedSemaphore
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
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
    train_dir = osp.join(dataset_dir, 'image_train')
    train_list = osp.join(dataset_dir, 'name_train.txt') # IMP Change
    test_dir = osp.join(dataset_dir,'image_test')
    test_list = osp.join(dataset_dir, 'name_test.txt')  # IMP Change
    query_dir = osp.join(dataset_dir, 'image_query')
    query_list = osp.join(dataset_dir, 'name_query.txt')
    img_train_path = osp.join(dataset_dir,'Dsl2/')
    img_test_path = osp.join(dataset_dir, 'Dsl2_test/')
    img_query_path = osp.join(dataset_dir,'Dsl2_query/')
    root_dir = osp.join(dataset_dir,'Dsl2')
    return V, dataset_dir, train_dir, train_list, test_dir, test_list,query_dir,query_list, img_train_path, img_test_path, img_query_path, root_dir

V,dataset_dir,train_dir,train_list,test_dir,test_list,query_dir,query_list,img_train_path,img_test_path,img_query_path,root_dir = directory_paths()

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

def save_pkl(D,path):
    with open(path, 'wb') as f:
        pickle.dump(D, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl_folder(Data, Data_Label, path,i):
    D = {}
    D['image'] = Data
    D['label'] = Data_Label
    tmp = path + f'{i}.pkl'
    save_pkl(D,tmp)

def get_data(train_dir,train_list,test_dir,test_list,query_dir,query_list,train_size,test_size,query_size):
    train_data = V.process_dir(train_dir,train_list, relabel=True)
    test_data = V.process_dir(test_dir,test_list, relabel=False)
    query_data = V.process_dir(query_dir,query_list, relabel=False)
    A = []
    for image in range(train_size):
        train_img = input_to_4d_tensor(train_data[image][0])
        train_label = train_data[image][1]
        save_pkl_folder(train_img,train_label,img_train_path,image)
    
    for image in range(test_size):
        test_img = input_to_4d_tensor(test_data[image][0])
        test_label = test_data[image][1]
        save_pkl_folder(test_img,test_label,img_test_path,image)

    for image in range(query_size):
        query_img = input_to_4d_tensor(query_data[image][0])
        query_label = query_data[image][1]
        A.append(query_label)
        save_pkl_folder(query_img,query_label,img_query_path,image)
    print(A)
    return train_data, test_data, query_data

train_data, test_data, query_data = get_data(train_dir,train_list,test_dir,test_list,query_dir,query_list,40,1000,28) #37746,11579,1678

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
            'label': torch.tensor(D['label']).clone().detach().type(torch.LongTensor)
        }

def data_loader(path,batch_size,b):
    veri = Veri(path)
    loader = DataLoader(veri, batch_size, shuffle=b)
    return loader,veri 



