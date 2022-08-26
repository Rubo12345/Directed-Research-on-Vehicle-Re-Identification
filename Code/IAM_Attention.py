
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax
import torchvision.transforms as transforms
from PIL import Image
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
T1 = time.time()

a = torch.randn((4,512,28,28),dtype = torch.float32)

def IAM_Attention(a):
    identity = a
    pd = (2,2,2,2)
    a = torch.nn.functional.pad(a, pd, mode='constant', value=0)
    m = nn.Softmax(dim = -1)
    for i in range(4):
        for c in range(512):
            for h in range(28):
                for w in range(28):
                    input = a[i,c,h:h+5,w:w+5]
                    output = m(input)
                    a[i,c,h:h+5,w:w+5] = output
    transform = transforms.CenterCrop((28,28))
    M = transform(a) 

    a = identity

    Lnuv = torch.tensor(-2)
    for i in range(4):
        for h in range(28):
            for w in range(28):
                m = a[i,0:,h,w]
                _,max_ = torch.max(m,0)
                Lnuv = a[i,max_,h,w]
                for c in range(512):
                    Lcuv = a[i,c,h,w]
                    G = Lcuv/Lnuv
                    a[i,c,h,w] = G
    G = a   

    Q = torch.multiply(M,G)
    Q_Dash = torch.max(Q[0:,0:,0:,0:],1)
    Q_Dash = Q_Dash.values.reshape((4,1,28,28))
    return Q_Dash
# print("Hi")
# F = IAM_Attention(a)
# print("Hi")
# print(F.shape)
# T2 = time.time()
# print("Time: ",(T2-T1))