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


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.orange = new_resnet.orange(4)
        self.green_red = new_resnet.Green_Red(576)
        self.purple = new_resnet.purple(4)
        self.blue = new_resnet.blue(576)
        self.iam = CAM_Module(Module) # use it as self.iam.forward(x)
        # self.class_name = class_name

    def forward_branch1(self, x):
        x = self.orange(x)
        out = self.purple(x)
        return out

    def forward_branch3(self, x):
        out = self.green_red(x)
        return out

    def forward_branch2(self, x):
        initial = x
        x = self.orange(x)
        x = self.iam.forward(x)
        y = Model.forward_branch3(self,initial)[3][1]
        m = torch.mul(x,y)
        out = self.blue(m)
        return out

    def forward(self, x):
        out1 = Model.forward_branch1(self,x)
        out2 = Model.forward_branch2(self,x)
        out3 = Model.forward_branch3(self,x)
        return out1, out2, out3

def the_model():
    model = Model()
    return model

def test():
    x = torch.randn(28,3,224,224)
    net = the_model()
    y = net(x)
    print(y[2][0].shape)
test()
