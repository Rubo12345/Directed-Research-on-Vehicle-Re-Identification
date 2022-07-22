import sys
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
import numpy as np
import torch
import pickle
from Datasets import Rotation, get_new_data
import ResNet
import os.path as osp
import xml.etree.ElementTree as ET

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def directory_paths():
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    Dsl_path = osp.join(dataset_dir,'Dsl/')
    Dsl_test_path = osp.join(dataset_dir, 'Dsl_test/')
    return dataset_dir,Dsl_path, Dsl_test_path

dataset_dir,Dsl_path, Dsl_test_path = directory_paths()
veri_loader, veri = get_new_data.data_loader(Dsl_path, 28)
veri_test_loader, veri_test = get_new_data.data_loader(Dsl_test_path,28)
class_names = veri.class_names

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
    resnet18_slb = ResNet.resnet18_SLB(4)  # Only four class 0,90,180,270
    resnet50_gb = ResNet.resnet50_bnneck_baseline(576) #    576 Vehicle class
    resnet18_gfb = ResNet.resnet18_GFB(576)
    loss_fn_1 = torch.nn.CrossEntropyLoss()
    loss_fn_2 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn_3 = torch.nn.TripletMarginLoss(margin=1,p=2)
    optimizer_1 = Optimizer('adam',resnet18_slb.parameters())
    optimizer_2 = Optimizer('adam',resnet18_gfb.parameters())
    optimizer_3 = Optimizer('adam',resnet50_gb.parameters())
    # optimizer = optimizer_1 + optimizer_2 + optimizer_3
    optimizer = Optimizer('adam',resnet18_gfb.parameters())          #doubt
    return resnet18_slb, resnet50_gb, resnet18_gfb, loss_fn_1, loss_fn_2, loss_fn_3, optimizer

resnet18_slb, resnet50_gb, resnet18_gfb, loss_fn_1, loss_fn_2, loss_fn_3, optimizer = model()

def train_slb(epochs):          #doubt for the training loop
    
    print('Start Training')
    
    for e in range(0, epochs):
        
        train_loss = 0; val_loss = 0
        
        resnet18_slb.train()
        resnet18_gfb.train()
        resnet50_gb.train()

        for train_step, dic in enumerate(veri_loader):

            train_images = dic['image'].squeeze()
            # print(train_images.shape)
            train_labels = dic['label'].squeeze()  # use the xml file
            # print(train_labels.shape)
            optimizer.zero_grad()                 # Zero the parameter gradient

            outputs_gfb = resnet18_gfb(train_images)  
            outputs_gb = resnet50_gb(train_images)

            '''
            Loss: Lambda(gb_tri)*L(gb_tri) + Lambda(gb_sce)*L(gb_sce) + Lambda(gfb_tri)*L(gfb_tri)
                    + Lambda(gfb_sce)*L(gfb_sce) + Lambda(slb)*L(slb)

            Lambda(gb_tri) = 0.5
            Lambda(gb_sce) = 0.5
            Lambda(gfb_tri) = 0.5
            Lambda(gfb_sce) = 0.5
            Lambda(slb) = 1.0
            '''

            # print(outputs_gb[1].shape)
            # print(outputs_gfb[1].shape)

            L_gb_tri = loss_fn_3(outputs_gb[1],train_labels)
            L_gb_sce = loss_fn_2(outputs_gb[0],train_labels)
            L_gfb_tri = loss_fn_3(outputs_gfb[1],train_labels)
            L_gfb_sce = loss_fn_2(outputs_gfb[0],train_labels)

            # For SLB Training   

            train_image = dic['image']
            R_0 = Rotation._apply_2d_rotation(train_image,0)
            R_90 = Rotation._apply_2d_rotation(train_image,90)
            R_180 = Rotation._apply_2d_rotation(train_image,180)
            R_270 = Rotation._apply_2d_rotation(train_image,270)
            Rot_Data = [R_0,R_90,R_180,R_270]
            Rot_Data_Label = ['0','1','2','3']
            L_slb = 0

            for i in range(len(Rot_Data)):
                outputs_slb = resnet18_slb(Rot_Data)
                L_slb += loss_fn_1(outputs_slb,Rot_Data_Label[i])

            # Total Loss

            loss = (0.5 * L_gb_tri) + (0.5*L_gb_sce) + (0.5*L_gfb_tri) + (0.5*L_gfb_sce) + (1*L_slb) 
            
            loss.backward()                       # Back Prop

            optimizer.step()                      # Adams Optimizer

            train_loss += loss.item()             # Train_loss Summation

            '''if train_step % 20 == 0:              # print every 20 train_steps

            #     print(f'[{e + 1}, {train_step + 1}] loss: {train_loss / 20:.3f}')
                
            #     correct = 0; n_samples = 0; accuracy = 0
                
            #     resnet18_slb.eval()
            #     resnet18_gfb.eval()
            #     resnet50_gb.eval()

            #     with torch.no_grad():
            #         for val_step, test_dic in enumerate(veri_test_loader):
                
            #             test_images = test_dic['image'].squeeze()
            #             test_labels = test_dic['label'].squeeze()
                        
            #             test_outputs_slb = resnet18_slb(test_images)
            #             test_outputs_gfb = resnet18_gfb(test_images)  
            #             test_outputs_gb = resnet50_gb(test_images)

            #             # loss = loss_fn(test_outputs, test_labels)
            #             L_gb_tri = loss_fn_3(test_outputs_gb[1],test_labels)
            #             L_gb_sce = loss_fn_2(test_outputs_gb[0],test_labels)
            #             L_gfb_tri = loss_fn_3(test_outputs_gfb[1],test_labels)
            #             L_gfb_sce = loss_fn_2(test_outputs_gfb[0],test_labels)
            #             L_slb = loss_fn_1(test_outputs_slb,test_labels)

            #             loss = (0.5 * L_gb_tri) + (0.5*L_gb_sce) + (0.5*L_gfb_tri) + (0.5*L_gfb_sce) + (1*L_slb) 

            #             val_loss += loss.item()

            #             # _, pred = torch.max(test_outputs,1)

            #             n_samples += test_labels.size(0)

            #             # correct += (pred == test_labels).sum().item()

            #         val_loss /= (val_step + 1)      

            #         # accuracy = 100 * correct / n_samples
                    
            #         # print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f} %')
            #         print(f'Validation Loss: {val_loss:.4f}')

            #     resnet18_slb.train()
            #     resnet18_gfb.train()
            #     resnet50_gb.train()'''

        train_loss /= (train_step + 1)

        print(f'Training Loss: {train_loss:.4f}')

    print("Training Finished")

train_slb(2) 

#
'''
1) model_REID(5images):
2) a = ResNet18()
3) a(input 4)
4) Loss-SLB
5) a(input 1) 
6) Loss - TRI - SCE
7) b = ResNet50()
8)  
'''