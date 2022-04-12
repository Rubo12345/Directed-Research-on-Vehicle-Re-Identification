from fileinput import filename
from multiprocessing import BoundedSemaphore
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    ''' Function converts the image into a tensor as well as size it'''
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = Image.open(I)
    Tensor = preprocess(input_image)
    Tensor = torch.reshape(Tensor,(1,3,Tensor.shape[1],Tensor.shape[2]))
    return Tensor

def Data_Rotation(Train_Images):
    Dsl = []
    for i in range(100):
        # image = mpimg.imread(Train_Images[i])
        _4d_tensor = input_to_4d_tensor(Train_Images[i])
        R_0 = Rotation._apply_2d_rotation(_4d_tensor,0)
        R_90 = Rotation._apply_2d_rotation(_4d_tensor,90)
        R_180 = Rotation._apply_2d_rotation(_4d_tensor,180)
        R_270 = Rotation._apply_2d_rotation(_4d_tensor,270)
        Dsl.append(R_0)
        Dsl.append(R_90)
        Dsl.append(R_180)
        Dsl.append(R_270)
    return Dsl

Train_Images, Train_Labels, Train_Cams = data_image_labels(train_dir, train_list)
Dsl = Data_Rotation(Train_Images)  # Now for loop for DSL

# input_batch = Dsl[0].unsqueeze(0)   #We don't need to do this as we already have a 4d_tensor.
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=False)
with torch.no_grad():
    output = model(Dsl[0])
probabilities = torch.nn.functional.softmax(output[0],dim=0)
print(probabilities)
print(max(probabilities))
a = max(probabilities)
print(list(probabilities).index(a))









'''#ResNet
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()
print(Train_Images[0])
filename = Train_Images[0]
filename = Dsl[0]
# input_image = Image.open(filename)
input_image = Dsl[0]
print(input_image.shape)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
print(input_batch.shape)

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)
print(len(probabilities))
'''