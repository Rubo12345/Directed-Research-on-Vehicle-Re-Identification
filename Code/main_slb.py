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
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
from Datasets import veri_train, Rotation

class SLB:
    def __init__(self):
        
        self.V = veri_train.VeRi()
        self.dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.train_list = None   # I think data gets shuffled using this  # 37778
        self.test_dir = osp.join(self.dataset_dir,'image_test')
        self.test_list = None
        self.Dsl_path = osp.join(self.dataset_dir,'Dsl')
        self.Dsl_test_path = osp.join(self.dataset_dir, 'Dsl_test')
        self.root_dir = osp.join(self.dataset_dir,'Dsl')
        
        self.veri = Veri(self.Dsl_path)
        self.veri_loader = torch.utils.data.DataLoader(self.veri, batch_size=28, shuffle=True)
        self.veri_test = Veri(self.Dsl_test_path)
        self.veri_test_loader = torch.utils.data.DataLoader(self.veri_test, batch_size=28, shuffle=True)


    def data_image_labels(self,train_dir, train_list):
        train_data = self.V.process_dir(train_dir,train_list, relabel=True)
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
            _4d_tensor = SLB.input_to_4d_tensor(Train_Images[i])
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

    Train_Images, Train_Labels, Train_Cams = data_image_labels(self.train_dir, self.train_list)
    Dsl, Dsl_Label= Data_Rotation(Train_Images,2000)
    Test_Images, Test_Labels, Test_Cams = data_image_labels(self.test_dir,self.test_list)
    Dsl_test, Dsl_Label_test = Data_Rotation(Test_Images,28)

    def save_pkl(D,path):
        with open(path, 'wb') as f:
            pickle.dump(D, f)

    def read_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    for i in range(len(Dsl)):
        D = {}
        D['image'] = Dsl[i]
        D['label'] = Dsl_Label[i]
        tmp = self.Dsl_path + f'{i}.pkl'
        save_pkl(D,tmp)

    for i in range(len(Dsl_test)):
        D_test = {}
        D_test['image'] = Dsl_test[i]
        D_test['label'] = Dsl_Label_test[i]
        tmp = self.Dsl_test_path + f'{i}.pkl'
        save_pkl(D,tmp)

    veri = Veri(Dsl_path)
    veri_loader = torch.utils.data.DataLoader(veri, batch_size=28, shuffle=True)

    veri_test = Veri(Dsl_test_path)
    veri_test_loader = torch.utils.data.DataLoader(veri_test, batch_size=28, shuffle=True)

    class_names = veri.class_names
    
    def show_images(images, labels,preds):
        plt.figure(figsize=(8, 4))
        for i, image in enumerate(images):
            plt.subplot(1, 28, i + 1, xticks=[], yticks=[])
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
            SLB.show_images(images_batch,labels_batch,labels_batch)
    # show_plot(veri_loader)

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=512, out_features=4)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

    def show_preds():
        resnet18.eval()
        for index, dic in enumerate(veri_loader):
            print(dic['image'].squeeze().size())
            images = dic['image'].squeeze()
            labels = dic['label'].squeeze()
            outputs = resnet18(images)
            _, preds = torch.max(outputs, 1)
            show_images(images, labels, preds)

    def train(epochs):
        print('Starting training..')
        for e in range(0, epochs):
            train_loss = 0.
            val_loss = 0.
            resnet18.train() # set model to training phase
            for train_step, dic in enumerate(veri_loader):
                optimizer.zero_grad()
                train_images = dic['image'].squeeze()
                train_labels = dic['label'].squeeze() 
                outputs = resnet18(train_images)
                loss = loss_fn(outputs, train_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                '''# if train_step % 20 == 0:
                #     print('Evaluating at step', train_step)
                #     accuracy = 0 
                #     resnet18.eval() # set model to eval phase
                #     for val_step, D_test in enumerate(veri_test_loader):
                #         test_images = D_test['image'].squeeze()
                #         test_labels = D_test['label'].squeeze()

                #         outputs = resnet18(test_images)
                #         loss = loss_fn(outputs, test_labels)
                #         val_loss += loss.item()

                #         _, preds = torch.max(outputs, 1)
                #         accuracy += sum((preds == test_labels).numpy())

                #     val_loss /= (val_step + 1)
                #     accuracy = accuracy/len(veri_test)
                #     print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                #     # show_preds()
                #     resnet18.train()
                #     # if accuracy >= 0.95:
                #     #     print('Performance condition satisfied, stopping..')
                #     #     return
            # train_loss /= (train_step +'' 1)'''

            print(f'Training Loss: {train_loss:.4f}')
        print('Training complete..')

    train(epochs=5)
    show_preds()

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
        D = SLB.read_pkl(file)
        return {
            'image': torch.tensor(D['image']),
            'label': torch.tensor(D['label'], dtype=torch.long),
            'class_names': self.class_names
        }

    def save_pkl(D,path):
        with open(path, 'wb') as f:
            pickle.dump(D, f)

    def read_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def store_data(Dsl,Dsl_Label,Dsl_path): # Can use this for test data also
        for i in range(len(Dsl)):
            D = {}
            D['image'] = Dsl[i]
            D['label'] = Dsl_Label[i]
            tmp = Dsl_path + f'{i}.pkl'
            Veri.save_pkl(D,tmp)

