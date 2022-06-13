
'''
    Self Supervised Learning
    Data -> Data Augumentation -> ResNet18 + 2 Basic Blocks, Complete -> loss -> back prop, optim 
'''

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
from Datasets import veri_train, Rotation
import ResNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def Data_Rotation(Train_Images,Data_Size):
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
    return Dsl, Dsl_Label

def get_data(No_of_Train_Images, No_of_Test_Images):
    Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
    Dsl, Dsl_Label= Data_Rotation(Train_Images,No_of_Train_Images)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(test_dir,test_list)
    Dsl_test, Dsl_Label_test = Data_Rotation(Test_Images,No_of_Test_Images)
    return Dsl, Dsl_Label, Dsl_test, Dsl_Label_test

Dsl, Dsl_Label, Dsl_test, Dsl_Label_test = get_data(560,56)  #4000,1120

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
save_pkl_folder(Dsl_test, Dsl_Label_test, Dsl_test_path)

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

veri_loader, veri = data_loader(Dsl_path, 28)
veri_test_loader, veri_test = data_loader(Dsl_test_path,28)
class_names = veri.class_names

def show_images(images, labels,preds):
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images):
        plt.subplot(1,28, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        plt.xlabel(f'{class_names[class_names.index(str(int(labels[i].numpy())))]}')
        plt.ylabel(f'{class_names[class_names.index(str(int(preds[i].numpy())))]}', color=col)
    plt.tight_layout()
    plt.show()

def show_plot(veri_loader):
    for index, dic in enumerate(veri_loader):
        # print(dic['image'].squeeze().size())
        images_batch = dic['image'].squeeze()
        labels_batch = dic['label'].squeeze()
        print(images_batch.shape)
        print(labels_batch.shape)
        show_images(images_batch,labels_batch,labels_batch)

def Optimizer(optim, param_groups):
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr=1e-4, weight_decay=5e-4,eps = 1e-8,
                                betas=(0.9,0.999))

    elif optim == 'amsgrad':
        return torch.optim.Adam(param_groups, lr=1e-4, weight_decay=5e-4,
                                betas=(0.9,0.999), amsgrad=True)

    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr=1e-4, momentum=0.9, weight_decay=5e-4,
                               dampening=0.0, nesterov=False)

    elif optim == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=1e-4, momentum=0.9, weight_decay=5e-4,
                                   alpha=0.99)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))

def model():
    # resnet18_slb = ResNet.resnet18_SLB(4).to(device)
    # resnet50_gb = ResNet.resnet50_bnneck_baseline(4).to(device)
    resnet18_GFB = ResNet.resnet18_GFB(4).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Optimizer('adam',resnet18_GFB.parameters())
    # optim = torch.optim.Adam(resnet18.parameters(), lr=1e-4, weight_decay=1e-8,betas=(0.9, 0.999))
    return resnet18_GFB, loss_fn, optimizer

resnet18_GFB, loss_fn, optimizer = model()

def show_preds(): 
    for index, dic in enumerate(veri_loader):
        images = dic['image'].squeeze()
        labels = dic['label'].squeeze()
        outputs = resnet50_gb(images)
        _, preds = torch.max(outputs, 1)
        show_images(images, labels, preds)

def train_slb(epochs):
    
    print('Start Training')
    
    for e in range(0, epochs):
        
        train_loss = 0; val_loss = 0
        
        for train_step, dic in enumerate(veri_loader):

            train_images = dic['image'].squeeze().to(device)

            train_labels = dic['label'].squeeze().to(device)
         
            optimizer.zero_grad()                 # Zero the parameter gradient

            outputs = resnet18_GFB(train_images)      # Fsl
            
            loss = loss_fn(outputs, train_labels) # Loss
            
            loss.backward()                       # Back Prop

            optimizer.step()                      # Adams Optimizer

            train_loss += loss.item()             # Train_loss Summation

            if train_step % 20 == 0:              # print every 20 train_steps
                
                print(f'[{e + 1}, {train_step + 1}] loss: {train_loss / 20:.3f}')
                
                correct = 0; n_samples = 0; accuracy = 0
                
                for val_step, test_dic in enumerate(veri_test_loader):
            
                    test_images = test_dic['image'].squeeze().to(device)

                    test_labels = test_dic['label'].squeeze().to(device)
            
                    test_outputs = resnet18_GFB(test_images)
            
                    loss = loss_fn(test_outputs, test_labels)

                    val_loss += loss.item()

                    _, pred = torch.max(test_outputs,1)

                    n_samples += test_labels.size(0)

                    correct += (pred == test_labels).sum().item()

                val_loss /= (val_step + 1)      

                accuracy = 100 * correct / n_samples
                
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f} %')

                # if accuracy >= 95:
                #     show_preds()

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')

    print("Training Finished")

train_slb(epochs=10)

