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
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
from Datasets import Rotation, get_new_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.orange = new_resnet.orange(4)
        self.green_red = new_resnet.Green_Red(575)
        self.purple = new_resnet.purple(4)
        self.blue = new_resnet.blue(575)
        # self.iam = CAM_Module(Module) # use it as self.iam.forward(x)
        self.iam = IAM_Attention
        self.loss_CE = torch.nn.CrossEntropyLoss()
        self.loss_CSE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_TRI = Losses.triplet_loss(margin=0.3)
        self.tri_label = torch.zeros(1,575) #training images
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward_branch1(self, x):
        R_0 = Rotation._apply_2d_rotation(x,0)
        R_90 = Rotation._apply_2d_rotation(x,90)
        R_180 = Rotation._apply_2d_rotation(x,180)
        R_270 = Rotation._apply_2d_rotation(x,270)
        Rot_Data = [R_0,R_90,R_180,R_270]
        Rot_Data_Label = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        Rot_Data_Label = torch.tensor(Rot_Data_Label, dtype = torch.long )
        # Rot_Data_Label = Rot_Data_Label.to(self.device)
        # L_slb = torch.tensor(0, dtype = torch.long).to(self.device)
        L_slb = torch.tensor(0)
        for i in range(len(Rot_Data)):
            torch.cuda.empty_cache()
            x = self.orange(Rot_Data[i])
            out = self.purple(x)
            L_slb = L_slb + self.loss_CE(out, Rot_Data_Label) 
        return out, L_slb

    def forward_branch3(self, x, y):
        out = self.green_red(x)
        # self.tri_label[0,y] = 1
        L_gb_tri = self.loss_TRI(out[1],y)
        L_gb_sce = self.loss_CSE(out[0], y)
        L_gb = L_gb_sce + L_gb_tri
        return out, L_gb

    def forward_branch2(self, x,y):
        initial = x
        x = self.orange(x)
        # x = self.iam.forward(x)
        x = self.iam.IAM_Attention(x)
        x1 = Model.forward_branch3(self,initial,y)[0][3][1]
        m = torch.mul(x,x1)
        out = self.blue(m)
        # self.tri_label[0,y] = 1
        L_gfb_tri = self.loss_TRI(out[1],y)
        L_gfb_sce = self.loss_CSE(out[0], y)
        L_gfb = L_gfb_sce + L_gfb_tri
        return out, L_gfb

    def forward(self, x,y):
        out1,L_slb = Model.forward_branch1(self,x)
        out3, L_gb = Model.forward_branch3(self,x,y)
        out2, L_gfb = Model.forward_branch2(self,x,y)
        return out1, L_slb, out2, L_gfb, out3, L_gb

def the_model():
    model = Model()
    return model

def test():
    x = torch.randn(28,3,224,224)
    Rot_Data_Label = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    y = torch.tensor(Rot_Data_Label)
    net = the_model()
    net = net
    f = net(x,y)
    print(f[0].shape)
    print(f[1])
    print(f[2][0].shape)
    print(f[3])
    print(f[4][0].shape)
    print(f[5])
test()
