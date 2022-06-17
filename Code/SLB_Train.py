import sys
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')

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
from Datasets import veri_train, Rotation, get_data
import ResNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def directory_paths():
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    Dsl_path = osp.join(dataset_dir,'Dsl/')
    Dsl_test_path = osp.join(dataset_dir, 'Dsl_test/')
    return dataset_dir,Dsl_path, Dsl_test_path

dataset_dir,Dsl_path, Dsl_test_path = directory_paths()
veri_loader, veri = get_data.data_loader(Dsl_path, 28)
veri_test_loader, veri_test = get_data.data_loader(Dsl_test_path,28)
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
    resnet18_slb = ResNet.resnet18_SLB(4)
    # resnet50_gb = ResNet.resnet50_bnneck_baseline(4).to(device)
    # resnet18_GFB = ResNet.resnet18_GFB(4)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = Optimizer('adam',resnet18_slb.parameters())
    # optim = torch.optim.Adam(resnet18.parameters(), lr=1e-4, weight_decay=1e-8,betas=(0.9, 0.999))
    return resnet18_slb, loss_fn, optimizer

resnet18_slb, loss_fn, optimizer = model()

def show_preds(): 
    for index, dic in enumerate(veri_loader):
        images = dic['image'].squeeze()
        labels = dic['label'].squeeze()
        outputs = resnet18_slb(images)
        _, preds = torch.max(outputs, 1)
        show_images(images, labels, preds)

def train_slb(epochs):
    
    print('Start Training')
    
    for e in range(0, epochs):
        
        train_loss = 0; val_loss = 0
        
        for train_step, dic in enumerate(veri_loader):

            train_images = dic['image'].squeeze()

            train_labels = dic['label'].squeeze()
        
            optimizer.zero_grad()                 # Zero the parameter gradient

            outputs = resnet18_slb(train_images)  # Fsl
            
            loss = loss_fn(outputs, train_labels) # Loss
            
            loss.backward()                       # Back Prop

            optimizer.step()                      # Adams Optimizer

            train_loss += loss.item()             # Train_loss Summation

            if train_step % 20 == 0:              # print every 20 train_steps

                print(f'[{e + 1}, {train_step + 1}] loss: {train_loss / 20:.3f}')
                
                correct = 0; n_samples = 0; accuracy = 0
                
                for val_step, test_dic in enumerate(veri_test_loader):
            
                    test_images = test_dic['image'].squeeze()

                    test_labels = test_dic['label'].squeeze()
            
                    test_outputs = resnet18_slb(test_images)
            
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

