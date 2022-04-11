# import tensorflow as tf

# # new_list=[145,56,89,56]
# # print(type(new_list))
# # con_lis = tf.convert_to_tensor(new_list)
# # print("Convert list to tensor:",con_lis)

# from PIL import Image
# im = Image.open("/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0496_c017_00035340_0.jpg")
# im = Image.open('/home/rutu/WPI/Directed_Research/ReID_Datasets/VeRi/image_train/0381_c004_00031195_0.jpg')
# # print(im)
# pix_val = list(im.getdata())
# # print(pix_val)
# # print(len(pix_val))  # length is 24336, but as it is RGB it should be 73008 
# pix_val_flat = [x for sets in pix_val for x in sets]
# # print(pix_val_flat)
# # print(len(pix_val_flat))   # length is 73008, flattened (Also check the sequence of colors R G B)

# import torch
# Tensor = torch.Tensor(pix_val_flat)
# Tensor = torch.reshape(Tensor, (1,3,156,156))
# print(Tensor)


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
# tensor([[[0.1922, 0.2275, 0.2157,  ..., 0.1569, 0.1608, 0.1804],
#          [0.1176, 0.1373, 0.1412,  ..., 0.2235, 0.2392, 0.2549],
#          [0.0824, 0.0863, 0.0941,  ..., 0.2706, 0.2824, 0.2863],
#          ...,
#          [0.4118, 0.4078, 0.4078,  ..., 0.1490, 0.1451, 0.1451],
#          [0.4157, 0.4157, 0.4157,  ..., 0.1569, 0.1569, 0.1569],
#          [0.4118, 0.4118, 0.4118,  ..., 0.1686, 0.1686, 0.1725]],

#         [[0.2039, 0.2392, 0.2275,  ..., 0.1686, 0.1725, 0.1882],
#          [0.1294, 0.1490, 0.1529,  ..., 0.2353, 0.2471, 0.2627],
#          [0.0941, 0.0980, 0.1059,  ..., 0.2784, 0.2863, 0.2902],
#          ...,
#          [0.4078, 0.4078, 0.4039,  ..., 0.1529, 0.1490, 0.1529],
#          [0.4157, 0.4118, 0.4118,  ..., 0.1608, 0.1608, 0.1647],
#          [0.4078, 0.4078, 0.4078,  ..., 0.1765, 0.1765, 0.1843]],

#         [[0.2235, 0.2588, 0.2471,  ..., 0.1882, 0.1922, 0.2078],
#          [0.1490, 0.1686, 0.1725,  ..., 0.2549, 0.2667, 0.2824],
#          [0.1137, 0.1176, 0.1255,  ..., 0.2980, 0.3059, 0.3098],
#          ...,
#          [0.4196, 0.4196, 0.4235,  ..., 0.1725, 0.1686, 0.1725],
#          [0.4314, 0.4314, 0.4314,  ..., 0.1804, 0.1804, 0.1843],
#          [0.4275, 0.4275, 0.4275,  ..., 0.1961, 0.1961, 0.2039]]])