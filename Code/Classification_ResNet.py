from __future__ import absolute_import
from __future__ import division
from cupshelpers import Device
from sklearn.metrics import log_loss

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet50_fc512']

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module,Conv2d,Parameter, Softmax
import torchvision
import torch.utils.model_zoo as model_zoo
from weight_init import init_pretrained_weights, weights_init_classifier,weights_init_kaiming
from attention import CAM_Module
import classifier
import time

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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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

class ResNet_18_Orange(nn.Module):
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
        super(ResNet_18_Orange, self).__init__()
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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) #error
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        return f

class Purple(nn.Module):
    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 512             # dimensions wanted
        super(Purple, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # purple network
        self.layer5 = self._make_layer(block, 512, layers[4], stride = 2) # BasicBlock
        self.layer6 = self._make_layer(block, 512, layers[5], stride = 2) # BasicBlock
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = classifier.classifier('cosine',512,4)
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

    def forward(self,x):
        x = self.layer5(x)
        x = self.layer6(x)
        v = self.global_avgpool(x)
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

class Blue(nn.Module):
    def __init__ (self, num_classes=575, loss={'softmax'}, pretrained=True, use_bnneck=True,
                 trans_classes=79, **kwargs):
        self.inplanes = 64
        super(Blue,self).__init__()
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

    def forward(self,x):
        x = self.global_branch[0](x)  #feature map
        x = self.global_branch[1](x)  #feature map
        global_feat = self.gap_global(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if not self.use_bnneck:
            bn_feat_global = global_feat
        elif self.use_bnneck:
            bn_feat_global = self.bottleneck_global(global_feat)  # normalize for angular softmax
        cls_score_global = self.classifier_global(bn_feat_global)
        return cls_score_global, global_feat , bn_feat_global

class ResNet50_Green_Red(nn.Module):
    def __init__(self, num_classes=576, loss={'softmax'}, pretrained=True, use_bnneck=True,
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

def orange(num_classes, loss = 'softmax', pretrained = True, **kwargs):
    model = ResNet_18_Orange(
        num_classes=num_classes,
        loss = loss,
        block = BasicBlock,
        layers = [2,2,2,2,2,2],  #according to the paper
        last_stride = 2,
        fc_dims = None,
        dropout_p = None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model,model_urls['resnet18'])
    return model

def purple(num_classes, loss = 'softmax', pretrained = True, **kwargs):
    model = Purple(
        num_classes=num_classes,
        loss = loss,
        block = BasicBlock,
        layers = [2,2,2,2,2,2], #according to the paper
        last_stride = 2,
        fc_dims = None,
        dropout_p = None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model,model_urls['resnet18'])
    return model

def blue(num_classes, loss={'softmax'}, pretrained=True, use_bnneck=True,
                **kwargs):
    model = Blue(
        num_classes=num_classes,
        loss=loss,
        # pretrained=pretrained,
        use_bnneck=use_bnneck,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def Green_Red(num_classes, loss={'softmax'}, pretrained=True,use_bnneck=True, **kwargs):
    model = ResNet50_Green_Red(
        num_classes=num_classes,
        loss=loss,
        # pretrained=pretrained,
        use_bnneck=use_bnneck,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def test():
    net1 = orange(4)
    net2 = purple(4)
    net3 = Green_Red(575)
    x = torch.randn(28,3,224,224)
    # x1 = net1(x)
    # x2 = net2(x1)
    F3T1 = time.time()
    for i in range(20):
        x = torch.randn(28,3,224,224)
        x3 = net3(x)
    F3T2 = time.time()
    print(x3[0].shape)
    print("F3 Time: ",(F3T2 - F3T1))
# test()

