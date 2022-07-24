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
sys.path.append('/home/rutu/WPI/Directed_Research/Directed-Research-on-Vehicle-Re-Identification/')
from Datasets import Rotation


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.orange = new_resnet.orange(4)
        self.green_red = new_resnet.Green_Red(576)
        self.purple = new_resnet.purple(4)
        self.blue = new_resnet.blue(576)
        self.iam = CAM_Module(Module) # use it as self.iam.forward(x)
        self.loss_CE = torch.nn.CrossEntropyLoss()
        self.loss_CSE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loss_TRI = torch.nn.TripletMarginLoss(margin=1,p=2)

    def forward_branch1(self, x):
        R_0 = Rotation._apply_2d_rotation(x,0) #check for input dimensions
        R_90 = Rotation._apply_2d_rotation(x,90)
        R_180 = Rotation._apply_2d_rotation(x,180)
        R_270 = Rotation._apply_2d_rotation(x,270)
        Rot_Data = [R_0,R_90,R_180,R_270]
        Rot_Data_Label = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        L_slb = 0

        for i in range(len(Rot_Data)):
            x = self.orange(Rot_Data[i])
            out = self.purple(x)
            # L_slb += 1
            Rot_Data_Label = torch.tensor(Rot_Data_Label)
            L_slb += self.loss_CE(out, Rot_Data_Label) # Compare the outputs
        return out, L_slb

    def forward_branch3(self, x):
        out = self.green_red(x)
        # L_gb = self.loss_CSE(out,)  # find labels for this
        L_gb_tri = 0
        L_gb_sce = 0
        L_gb = L_gb_sce + L_gb_tri
        return out, L_gb

    def forward_branch2(self, x):
        initial = x
        x = self.orange(x)
        x = self.iam.forward(x)
        y = Model.forward_branch3(self,initial)[0][3][1]
        m = torch.mul(x,y)
        out = self.blue(m)
        L_gfb_tri = 0
        L_gfb_sce = 0
        L_gfb = L_gfb_sce + L_gfb_tri
        return out, L_gfb

    def forward(self, x):
        out1,L_slb = Model.forward_branch1(self,x)
        out2, L_gfb = Model.forward_branch2(self,x)
        out3, L_gb = Model.forward_branch3(self,x)
        return out1, L_slb, out2, L_gfb, out3, L_gb

def the_model():
    model = Model()
    return model

def test():
    x = torch.randn(28,3,224,224)
    net = the_model()
    y = net(x)
    print(y[0][0].shape)
test()
