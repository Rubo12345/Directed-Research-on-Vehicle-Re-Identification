from __future__ import absolute_import
from __future__ import division

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet50_fc512']

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module,Conv2d,Parameter, Softmax
import torchvision
import torch.utils.model_zoo as model_zoo
from weight_init import init_pretrained_weights, weights_init_classifier,weights_init_kaiming
from attention import CAM_Module
import logging
import math
import new_resnet


class Model():
    def __init__(self):
        super(Model,self).__init__()
        self.orange = new_resnet.orange()
        self.green_red = new_resnet.Green_Red()
        self.purple = new_resnet.purple()
        self.blue = new_resnet.blue()
        self.iam = CAM_Module(Module) # use it as self.iam.forward(x)

    def forward_branch1(self, x):
        x = self.orange(4)
        out = self.purple(x)
        return out

    def forward_branch3(self, x):
        out = self.green_red(x)
        return out[0],out[2]

    def forward_branch2(self, x):
        initial = x
        x = self.orange(x)
        x = self.iam.forward(x)
        y = Model.forward_branch3(initial)
        m = torch.mul(x,y)
        out = self.blue(m)
        return out

    # def forward(self, img):
        
