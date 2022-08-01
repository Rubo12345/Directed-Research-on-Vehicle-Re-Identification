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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """Residual network.

    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])              # Conv_2x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # Conv_3x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)   # Conv_4x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)   # Conv_5x

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.classifier = nn.CosineSimilarity(dim = self.feature_dim, eps = 1e-8)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_param(self, name):
        param_dict = model_zoo.load_url(model_urls[name])
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def featuremaps(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet_SLB(nn.Module):
    """Residual network.

    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet_SLB, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])              # Conv_2x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # Conv_3x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)   # Conv_4x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)   # Conv_5x

        ''' Basic Block Layers for SLB'''
        self.layer5 = self._make_layer(block, 512, layers[4], stride = 2) # BasicBlock
        self.layer6 = self._make_layer(block, 512, layers[5], stride = 2) # BasicBlock
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)
        x = self.layer6(x)
        # print(x.shape)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet_GFB(nn.Module):
    """Residual network.

    Reference:
        He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 use_bnneck = True,
                 **kwargs):
        self.inplanes = 64
        super(ResNet_GFB, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])              # Conv_2x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # Conv_3x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)   # Conv_4x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)   # Conv_5x

        #---------------------------------------------
        base = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            with_classifier=False,
            fc_dims=None,
            dropout_p=None,
            **kwargs)
        base.load_param('resnet50')

        self.common_out_dim = 2048
        self.shallow_branch = nn.Sequential(base.conv1, base.bn1, base.relu,base.maxpool, base.layer1, base.layer2)
        self.global_branch = nn.Sequential(base.layer3, base.layer4)
        self.gap_global = nn.AdaptiveAvgPool2d(1)
        self.use_bnneck = use_bnneck

        if not self.use_bnneck:
            self.classifier_global = nn.Linear(self.common_out_dim, num_classes)
        elif self.use_bnneck:
            self.bottleneck_global = nn.BatchNorm1d(self.common_out_dim)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.classifier_global = nn.Linear(self.common_out_dim, num_classes, bias=False)

            self.bottleneck_global.apply(weights_init_kaiming)
            self.classifier_global.apply(weights_init_classifier)

        channels = [256, 512, 1024, 2048]
        self.domain_norms = nn.ModuleList()
        for index in range(len(channels)):
            self.domain_norms.append(IBN(channels[index]))

        '''# if self.args.IBN_OR_IN == 'IN':
        #     for index in range(len(channels)):
        #         self.domain_norms.append(nn.InstanceNorm2d(channels[index], affine=True))
        # elif self.args.IBN_OR_IN == 'IBN':
        #     for index in range(len(channels)):
        #         self.domain_norms.append(IBN(channels[index]))
        # else:
        #     raise IOError'''

        #---------------------------------------------

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_param(self, name):
        param_dict = model_zoo.load_url(model_urls[name])
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def featuremaps(self, x):
        # torch.cuda.empty_cache()
        Initial = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        x1 = CAM_Module(Module).forward(x)
        
        # print(x.shape)
        
        net = resnet50_bnneck_baseline(4)
        
        y,v,bn_v,layers = net(Initial)

        RN50_layer2 = layers[1] # layer2

        # print(RN50_layer2.shape)
        
        m = torch.mul(x1,RN50_layer2)    # Element-Wise Multiplication
        
        # print(m.shape)
        
        f_layer3 = self.global_branch[0](m)
        
        # print(f_layer3.shape)
        
        f_layer4 = self.global_branch[1](f_layer3)
        
        # print(f_layer4.shape)
        
        return f_layer4

    def forward(self, x):
        f = self.featuremaps(x)
        global_feat = self.global_avgpool(f)
        # print(global_feat.shape)
        global_feat = global_feat.view(global_feat.size(0), -1)
        # print(global_feat.shape)

        if not self.use_bnneck:
            bn_feat_global = global_feat
        elif self.use_bnneck:
            bn_feat_global = self.bottleneck_global(global_feat)  # normalize for angular softmax

        cls_score_global = self.classifier_global(bn_feat_global)
        # print(cls_score_global.shape)
        # print(cls_score_global.shape)
        # return cls_score_global, global_feat, None, None  # global feature for triplet loss
        # return cls_score_global, global_feat, bn_feat_global, [f_layer1, f_layer2, f_layer3, f_layer4] # global feature for triplet lossd
        return cls_score_global, global_feat       

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class ResNet50_BNNeck_baseline(nn.Module):
    def __init__(self, num_classes=576, loss={'xent'}, pretrained=True, use_bnneck=True,
                 trans_classes=79, **kwargs):
        super().__init__()
        base = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            with_classifier=False,
            fc_dims=None,
            dropout_p=None,
            **kwargs)
        base.load_param('resnet50')
                
        self.common_out_dim = 2048
        self.shallow_branch = nn.Sequential(base.conv1, base.bn1, base.relu,base.maxpool, base.layer1, base.layer2)
        self.global_branch = nn.Sequential(base.layer3, base.layer4)
        self.gap_global = nn.AdaptiveAvgPool2d(1)
        self.use_bnneck = use_bnneck

        if not self.use_bnneck:
            self.classifier_global = nn.Linear(self.common_out_dim, num_classes)
        elif self.use_bnneck:
            self.bottleneck_global = nn.BatchNorm1d(self.common_out_dim)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.classifier_global = nn.Linear(self.common_out_dim, num_classes, bias=False)

            self.bottleneck_global.apply(weights_init_kaiming)
            self.classifier_global.apply(weights_init_classifier)

        channels = [256, 512, 1024, 2048]
        self.domain_norms = nn.ModuleList()
        for index in range(len(channels)):
            self.domain_norms.append(IBN(channels[index]))

        '''# if self.args.IBN_OR_IN == 'IN':
        #     for index in range(len(channels)):
        #         self.domain_norms.append(nn.InstanceNorm2d(channels[index], affine=True))
        # elif self.args.IBN_OR_IN == 'IBN':
        #     for index in range(len(channels)):
        #         self.domain_norms.append(IBN(channels[index]))
        # else:
        #     raise IOError'''

    def forward(self, x):
        #feature map
        f_layer1 = self.shallow_branch[:-1](x)
        f_layer2 = self.shallow_branch[-1](f_layer1)
        f_layer3 = self.global_branch[0](f_layer2)
        f_layer4 = self.global_branch[1](f_layer3)
        f_g = f_layer4
        global_feat = self.gap_global(f_g)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if not self.use_bnneck:
            bn_feat_global = global_feat
        elif self.use_bnneck:
            bn_feat_global = self.bottleneck_global(global_feat)  # normalize for angular softmax

        cls_score_global = self.classifier_global(bn_feat_global)
        # return cls_score_global, global_feat, None, None  # global feature for triplet loss
        # return cls_score_global, global_feat, bn_feat_global, [f_layer1, f_layer2, f_layer3, f_layer4] # global feature for triplet lossd
        # return cls_score_global, global_feat, bn_feat_global # global feature for triplet lossds
        # return cls_score_global, f_layer2
        return cls_score_global, global_feat, bn_feat_global, [f_layer1, f_layer2, f_layer3, f_layer4] # global feature for triplet lossd

"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""

def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def resnet18_GFB(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet_GFB(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def resnet18_SLB(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = ResNet_SLB(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2, 2, 2],  #according to the paper
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model

def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    
    # print(model)
    return model

def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model

def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model

def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    """
    resnet + fc
    """
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet50_bnneck_baseline(num_classes, loss={'softmax'}, pretrained=True, **kwargs):
    model = ResNet50_BNNeck_baseline(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs
    )
    return model

def test():
    net = resnet18_GFB(4)
    x = torch.randn(28,3,224,224)
    # y = net(x).to('cuda')
    y = net(x)[0].to(device) 
    print(y.shape)
# test()

