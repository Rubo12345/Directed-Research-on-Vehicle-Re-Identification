import sys
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
import numpy as np
import torch
from Datasets import get_new_data
import os.path as osp
import xml.etree.ElementTree as ET
import model
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import glob
import os

PATH = '/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/model_weights.pth'

def directory_paths():
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    img_train_path = osp.join(dataset_dir,'Dsl/')
    img_test_path = osp.join(dataset_dir, 'Dsl_test/')
    img_query_path = osp.join(dataset_dir,'Dsl_query/')
    return dataset_dir,img_train_path, img_test_path,img_query_path

dataset_dir,img_train_path, img_test_path,img_query_path = directory_paths()
# veri_loader, veri = get_new_data.data_loader(img_train_path,4,True)
veri_test_loader, veri_test = get_new_data.data_loader(img_test_path,4,False)
veri_query_loader, veri_query = get_new_data.data_loader(img_query_path,4,False)
# print(len(veri_query_loader))

def show_images(images, labels,preds):
    plt.figure(figsize=(80, 40))
    for i, image in enumerate(images):
        plt.subplot(1,4, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        # plt.xlabel(f'{class_names[class_names.index(str(int(labels[i].numpy())))]}')
        # plt.ylabel(f'{class_names[class_names.index(str(int(preds[i].numpy())))]}', color=col)
    plt.tight_layout()
    plt.show()

def show_plot(loader):
    for index, dic in enumerate(loader):
        images_batch = dic['image'].squeeze()
        labels_batch = dic['label'].squeeze()
        # print(images_batch.shape)
        print(labels_batch)
        show_images(images_batch,labels_batch,labels_batch)

# show_plot(veri_query_loader)

def Optimizer(optim, param_groups, lr):
    if optim == 'adam':
        return torch.optim.Adam(param_groups, lr, weight_decay=5e-4,eps = 1e-8,
                                betas=(0.9,0.999))

    elif optim == 'amsgrad':
        return torch.optim.Adam(param_groups, lr, weight_decay=5e-4,
                                betas=(0.9,0.999), amsgrad=True)

    elif optim == 'sgd':
        return torch.optim.SGD(param_groups, lr, momentum=0.9, weight_decay=5e-4,
                               dampening=0.0, nesterov=False)

    elif optim == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr, momentum=0.9, weight_decay=5e-4,
                                   alpha=0.99)

    else:
        raise ValueError('Unsupported optimizer: {}'.format(optim))

def Model():
    the_model = model.the_model()
    optimizer = Optimizer('adam',the_model.parameters(),lr = 1e-4)          #doubt
    return the_model, optimizer

the_model,optimizer = Model()
the_model.to(device)

the_model.load_state_dict(torch.load(PATH))
the_model.eval()

def test():
    correct = 0
    '''slb_op = [];gfb_op = []; gb = []
    testing_loss = 0
    c1 = "SLB_OUTPUT"
    c2 = "GFB_OUTPUT"
    c3 = "GB_OUTPUT"
    Acc = []'''
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print("Start Testing: ")
    with torch.no_grad():
        # print("Hi-1")
        for query_step, query_dic in enumerate(veri_query_loader):
            # print("Hi-2")
            rank_list_1 = []; rank_list_2 = []; rank_list_3 = []
            query_images = query_dic['image'].squeeze().to(device)
            query_labels = query_dic['label'].squeeze().type(torch.LongTensor).to(device)
            print(query_labels)
            query_output = the_model(query_images,query_labels)
            
            '''# query_SLB = query_output[0]  # check the feature map if needed
            # slb_op.append(query_SLB)
            # query_GFB = query_output[2][0]
            # slb_op.append(query_GFB)
            # query_GB = query_output[5][0]
            # slb_op.append(query_GB)'''
            
            query_GFB = query_output[2][1]

            for i in range(4):
                # print("Hi-3")
                query_gfb = query_GFB[i]
                query_gfb = query_gfb.reshape((1,2048))
                Label = query_labels[i]
                print(Label)
                rank_list_2 = []

                for test_step, test_dic in enumerate(veri_test_loader):
                    # print("Hi-4")
                    test_images = test_dic['image'].squeeze().to(device)
                    test_labels = test_dic['label'].squeeze().type(torch.LongTensor).to(device)
                    test_output = the_model(test_images,test_labels)

                    '''# test_SLB = test_output[0]
                    # test_GFB = test_output[2][0]
                    # test_GB = test_output[5][0]

                    # CC1 = nn.CosineSimilarity(test_SLB,query_SLB)
                    # CC2 = nn.CosineSimilarity(test_GFB,query_GFB)
                    # CC3 = nn.CosineSimilarity(test_GB,query_GB)

                    # rank_list_1.append(CC1)
                    # rank_list_2.append(CC2)
                    # rank_list_3.append(CC3)'''

                    test_GFB = test_output[2][1]

                    for j in range(4):
                        # print("Hi-5")
                        test_gfb = test_GFB[j]
                        test_gfb = test_gfb.reshape((1,2048))
                        CC2 = cos(test_gfb,query_gfb)
                        rank_list_2.append(CC2)

                a = rank_list_2.index(max(rank_list_2))
                print(a)
                pred = get_new_data.test_data[a][1]  
                pred = torch.tensor(pred).to(device)
                print(pred)
                
                '''#print(" ")
                L_slb = test_output[1]
                # print("L_slb: ",L_slb)
                L_gfb = test_output[3]
                # print("L_gfb: ",L_gfb)
                L_gb = test_output[5]
                # print("L_gb: ",L_gb)
                # print(" ")

                loss = (0.5 * L_gfb) + (0.5 * L_gb) + L_slb 

                testing_loss += loss.item()

                _, pred = test_output[2][0].max(1)
                
                print(" ")
                print("Prediction: ",pred)
                print("Test_labels: ",test_labels)
                print(" ")
                '''

                correct += (pred == Label).sum().item()
                print(" ")
                print(correct)
                print(" ")

    n_samples = 28

    accuracy = 100 * correct / n_samples

    print(" ")
    print(f'Accuracy: {accuracy:.4f} %')
    print(" ")

    files_1 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl/*')
    files_2 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl_test/*')
    files_3 = glob.glob('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/Dsl_query/*')
    for f in files_1:
        os.remove(f)
    for g in files_2:
        os.remove(g)
    for h in files_3:
        os.remove(h)

test()     