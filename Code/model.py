from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet50_fc512']

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module,Conv2d,Parameter, Softmax
import torch.utils.model_zoo as model_zoo
from weight_init import init_pretrained_weights, weights_init_classifier,weights_init_kaiming
from attention import CAM_Module
import new_resnet
import sys
import Losses
import IAM_Attention
import time
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
from Datasets import Rotation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.orange = new_resnet.orange(4)
        self.green_red = new_resnet.Green_Red(575)
        self.purple = new_resnet.purple(4)
        self.blue = new_resnet.blue(575)
        self.iam = CAM_Module(Module) # use it as self.iam.forward(x)
        # self.iam = IAM_Attention
        self.loss_CE = torch.nn.CrossEntropyLoss()
        self.loss_CSE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_TRI = Losses.triplet_loss(margin=0.3)
        self.tri_label = torch.zeros(1,575) #training images
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward_branch1(self, x):
        F1T1 = time.time()
        R_0 = Rotation._apply_2d_rotation(x,0)
        R_90 = Rotation._apply_2d_rotation(x,90)
        R_180 = Rotation._apply_2d_rotation(x,180)
        R_270 = Rotation._apply_2d_rotation(x,270)
        Rot_Data = [R_0,R_90,R_180,R_270]
        Rot_Data_Label_0 = [0,0,0,0]
        Rot_Data_Label_90 = [1,1,1,1]
        Rot_Data_Label_180 = [2,2,2,2]
        Rot_Data_Label_270 = [3,3,3,3]
        Rot_Data_Label = [Rot_Data_Label_0,Rot_Data_Label_90,Rot_Data_Label_180,Rot_Data_Label_270]
        Rot_Data_Label = torch.tensor(Rot_Data_Label,dtype = torch.long)
        Rot_Data_Label = Rot_Data_Label.type(torch.LongTensor).to(device)
        L_slb = torch.tensor(0, dtype = torch.long).to(device)
        # L_slb = torch.tensor(0, dtype = torch.long)
        next = []
        for i in range(len(Rot_Data)):
            x = Rot_Data[i].to(device)
            x = self.orange(x)
            next.append(x)
            out = self.purple(x)
            L_slb = L_slb + self.loss_CE(out, Rot_Data_Label[i]) 
        F1T2 = time.time()
        return out, L_slb, next[0]

    def forward_branch3(self, x, y):
        F3T1 = time.time()
        out = self.green_red(x)
        L_gb_tri = self.loss_TRI(out[1],y)
        L_gb_sce = self.loss_CSE(out[0], y)
        L_gb = L_gb_sce + L_gb_tri
        F3T2 = time.time()
        return out, L_gb

    def forward_branch2(self,x,y,o1):
        F2T1 = time.time()
        initial = x
        x = self.iam.forward(o1)
        # x = self.iam.IAM_Attention(o1)
        F3T1 = time.time()
        GB_out,L_gb = Model.forward_branch3(self,initial,y)
        F3T2 = time.time()
        x1 = GB_out[3][1]
        m = torch.mul(x,x1)
        out = self.blue(m)
        L_gfb_tri = self.loss_TRI(out[1],y)
        L_gfb_sce = self.loss_CSE(out[0], y)
        L_gfb = L_gfb_sce + L_gfb_tri
        F2T2 = time.time()
        return out, L_gfb, GB_out, L_gb

    def forward(self, x,y):
        out1,L_slb,o1 = Model.forward_branch1(self,x)
        # out3, L_gb = Model.forward_branch3(self,x,y)
        out2, L_gfb, out3, L_gb = Model.forward_branch2(self,x,y,o1)
        return out1, L_slb, out2, L_gfb, out3, L_gb

def the_model():
    model = Model()
    return model

def test():
    T1 = time.time()
    x = torch.randn(4,3,224,224).to(device)
    # Rot_Data_Label = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    Rot_Data_Label = [0,1,2,3]
    y = torch.tensor(Rot_Data_Label).to(device)
    net = the_model().to(device)
    net = net
    f = net(x,y)
    print(f[0].shape)
    print(f[1])
    print(f[2][0].shape)
    print(f[3])
    print(f[4][0].shape)
    print(f[5])
    T2 = time.time()
    print("Time",(T2-T1))
# test()
