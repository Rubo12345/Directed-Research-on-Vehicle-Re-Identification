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

PATH = '/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/model_weights.pth'

def directory_paths():
    dataset_dir = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi'
    img_train_path = osp.join(dataset_dir,'Dsl2/')
    img_test_path = osp.join(dataset_dir, 'Dsl2_test/')
    img_query_path = osp.join(dataset_dir,'Dsl2_query/')
    return dataset_dir,img_train_path, img_test_path,img_query_path

dataset_dir,img_train_path, img_test_path,img_query_path = directory_paths()
# veri_loader, veri = get_new_data.data_loader(img_train_path,4,True)
veri_test_loader, veri_test = get_new_data.data_loader(img_test_path,4,False)
veri_query_loader, veri_query = get_new_data.data_loader(img_query_path,4,False)

def show_images(images, labels,preds):
    plt.figure(figsize=(8, 4))
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
        print(images_batch.shape)
        print(labels_batch)
        show_images(images_batch,labels_batch,labels_batch)

'''for val_step, test_dic in enumerate(veri_loader):     
    test_images = test_dic['image'].squeeze().to(device)
    test_labels = test_dic['label'].squeeze().type(torch.LongTensor).to(device)
    print(test_labels)
show_plot(veri_test_loader)'''

def save_checkpoint(state,filename = "my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state,filename)

def lr_scheduler(epochs,lr):
    if epochs >= 60:
        lr = lr/10
    elif epochs >= 40:
        lr = lr/10
    elif epochs >= 20:
        lr = lr /10
    else:
        lr = lr
    return lr

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

# def train_slb(epochs):          
    
#     T1 = time.time()
#     print('Start Training')
    
#     col1 = "sb"
#     col2 = "gfb"
#     col3 = "gb"
#     col4 = "Total Loss"
#     col5 = "Training Loss"

#     slb = []; gfb = []; gb = []
#     Tr_Loss = [];Loss = []

#     for e in range(0, epochs):

#         train_loss = 0; val_loss = 0; n_samples = 0; correct = 0

#         check_point = {'state_dict':the_model.state_dict(),'optimizer':optimizer.state_dict()}
#         save_checkpoint(check_point)

#         the_model.train()

#         for train_step, train_dic in enumerate(veri_loader):
            
#             train_images = train_dic['image'].squeeze().to(device)
#             train_labels = train_dic['label'].squeeze().type(torch.LongTensor).to(device)

#             optimizer.zero_grad()  
                  
#             output = the_model(train_images,train_labels)

#             '''
#             Loss: Lambda(gb_tri)*L(gb_tri) + Lambda(gb_sce)*L(gb_sce) + Lambda(gfb_tri)*L(gfb_tri)
#                     + Lambda(gfb_sce)*L(gfb_sce) + Lambda(slb)*L(slb)

#             Lambda(gb_tri) = 0.5
#             Lambda(gb_sce) = 0.5
#             Lambda(gfb_tri) = 0.5
#             Lambda(gfb_sce) = 0.5
#             Lambda(slb) = 1.0
#             '''

#             print(" ")
#             L_slb = output[1]
#             slb.append(float(L_slb))
#             print(L_slb)
#             L_gfb = output[3]
#             gfb.append(float(L_gfb))
#             print(L_gfb)
#             L_gb = output[5]
#             gb.append(float(L_gb))
#             print(L_gb)
#             print(" ")

#             loss = (0.5 * L_gfb) + (0.5 * L_gb) + L_slb 

#             Loss.append(float(loss))

#             loss.backward()                       
            
#             optimizer.step()              
            
#             print("[ Epoch:", e , " , train_step: ",train_step," , loss: ",loss.item(),"]")
            
#             train_loss += loss.item()            

#             # if train_step % 10 == 0:              
                
#             #     correct = 0; n_samples = 0; accuracy = 0
                
#             #     the_model.eval()

#             #     with torch.no_grad():
#             #         for query_step, query_dic in enumerate(veri_query_loader):
#             #             rank_list = []
#             #             query_images = query_dic['image'].squeeze().to(device)
#             #             query_images = query_images.reshape(1,3,224,224)
#             #             query_labels = query_dic['label'].squeeze().type(torch.LongTensor).to(device)
                        
#             #             query_output = the_model(query_images,query_labels)
                        
#             #             for test_step, test_dic in enumerate(veri_test_loader):
                    
#             #                 test_images = test_dic['image'].squeeze().to(device)
#             #                 test_images = test_images.reshape(1,3,224,224)
#             #                 test_labels = test_dic['label'].squeeze().type(torch.LongTensor).to(device)

#             #                 test_output = the_model(test_images,test_labels)

#             #                 CC = nn.CosineSimilarity(test_output[2][0],query_output[2][0])

#             #                 rank_list.append(CC)
                            
#             #                 a = rank_list.index(max(rank_list))

#             #                 print(len(rank_list))
                
#             #     the_model.train()
    
#         train_loss /= (train_step + 1)
#         Tr_Loss.append(train_loss)
        
#         print(" ")
#         print(f'Training Loss: {train_loss:.4f}')
#         print(" ")

#     print("Training Finished")
#     torch.save(the_model.state_dict(),'model_weights.pth')
#     data = pd.DataFrame({col1:slb,col2:gfb,col3:gb,col4:Loss})
#     data.to_excel('Losses_1.xlsx',sheet_name = 'Compare_Losses', index = True)
#     data2 = pd.DataFrame({col5:Tr_Loss})
#     data2.to_excel('Losses_2.xlsx',sheet_name = 'Compare_Losses', index = True)
#     T2 = time.time()
#     print("Time",(T2-T1))
# # train_slb(10) 

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
    with torch.no_grad():
        for query_step, query_dic in enumerate(veri_query_loader):
            rank_list_1 = []; rank_list_2 = []; rank_list_3 = []
            query_images = query_dic['image'].squeeze().to(device)
            query_labels = query_dic['label'].squeeze().type(torch.LongTensor).to(device)
            query_output = the_model(query_images,query_labels)
            
            '''# query_SLB = query_output[0]  # check the feature map if needed
            # slb_op.append(query_SLB)
            # query_GFB = query_output[2][0]
            # slb_op.append(query_GFB)
            # query_GB = query_output[5][0]
            # slb_op.append(query_GB)'''
            
            query_GFB = query_output[2][0]
          
            for i in range(4):
                query_gfb = query_GFB[i]
                query_gfb = query_gfb.reshape((1,575))
                Label = query_labels[i]
                rank_list_2 = []
                for test_step, test_dic in enumerate(veri_test_loader):
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

                    test_GFB = test_output[2][0]
                    
                    for j in range(4):
                        test_gfb = test_GFB[j]
                        test_gfb = test_gfb.reshape((1,575))
                        CC2 = cos(test_gfb,query_gfb)
                        rank_list_2.append(CC2)
                print(len(rank_list_2))
                print(rank_list_2)
                a = rank_list_2.index(max(rank_list_2))
            
                pred = get_new_data.test_data[a][1]  
                
                '''# print(" ")
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
                print(" ")'''

                # n_samples += query_labels.size(0)

                correct += (pred == Label).sum().item()
                print(" ")
                print(correct)
                print(" ")
        n_samples = 160

        accuracy = 100 * correct / n_samples

        print(" ")
        print(f'Accuracy: {accuracy:.4f} %')
        print(" ")

test()     