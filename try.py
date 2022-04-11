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

image = '/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0435_c019_00023560_0.jpg'
input_image = Image.open(image)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
print(input_tensor)
print(input_tensor.shape)
input_tensor = torch.reshape(input_tensor,(1,3,input_tensor.shape[1],input_tensor.shape[2]))   
print(input_tensor.shape)
