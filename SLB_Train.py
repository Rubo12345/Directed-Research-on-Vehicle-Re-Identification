import numpy as np
import torch
import os.path as osp
from PIL import Image
import sys
sys.path.append('/home/rutu/WPI/Directed_Research/My_Approach_To_DR') 
from Datasets import veri_train, Rotation

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
    Tensor = torch.Tensor(I)
    # Tensor = torch.reshape(Tensor, (1,3,156,156))
    Tensor = torch.reshape(Tensor,(1,3,int(np.sqrt(len(I)/3)),int(np.sqrt(len(I)/3))))   #check this
    return Tensor

def Data_Rotation(Train_Images):
    Dsl = []
    for i in range(1):
        image = Image.open(Train_Images[i])
        # image = Image.open("/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0496_c017_00035340_0.jpg")
        image_to_pixel = image_to_pixel(image)
        input_to_4d_tensor = input_to_4d_tensor(image_to_pixel)
        R_0 = Rotation._apply_2d_rotation(input_to_4d_tensor,0)
        R_90 = Rotation._apply_2d_rotation(input_to_4d_tensor,90)
        R_180 = Rotation._apply_2d_rotation(input_to_4d_tensor,180)
        R_270 = Rotation._apply_2d_rotation(input_to_4d_tensor,270)
        Dsl.append(R_0)
        Dsl.append(R_90)
        Dsl.append(R_180)
        Dsl.append(R_270)
    return Dsl

Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)

# image = Image.open("/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0496_c017_00035340_0.jpg")
# # print(image_to_pixel(image))
# I = image_to_pixel(image)
# I = input_to_4d_tensor(I)
# R = Rotation._apply_2d_rotation(I,90)
# print(R)

Dsl = []
for i in range(1):
    # image = Image.open(Train_Images[i])
    image = Image.open("/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0496_c017_00035340_0.jpg")
    image_to_pixel = image_to_pixel(image)
    input_to_4d_tensor = input_to_4d_tensor(image_to_pixel)
    R_0 = Rotation._apply_2d_rotation(input_to_4d_tensor,0)
    R_90 = Rotation._apply_2d_rotation(input_to_4d_tensor,90)
    R_180 = Rotation._apply_2d_rotation(input_to_4d_tensor,180)
    R_270 = Rotation._apply_2d_rotation(input_to_4d_tensor,270)
    Dsl.append(R_0)
    Dsl.append(R_90)
    Dsl.append(R_180)
    Dsl.append(R_270)
print(Dsl)

