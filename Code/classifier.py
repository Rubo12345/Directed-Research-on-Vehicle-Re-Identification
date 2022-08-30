import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_fully_connected_layer(
    x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True
):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out

def global_pooling(x, pool_type):
    assert x.dim() == 4
    if pool_type == "max":
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif pool_type == "avg":
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError("Unknown pooling type.")

class GlobalPooling(nn.Module):
    def __init__(self, pool_type):
        super().__init__()
        assert pool_type == "avg" or pool_type == "max"
        self.pool_type = pool_type

    def forward(self, x):
        return global_pooling(x, pool_type=self.pool_type)

class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=20.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )

    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )
        learnable = self.scale_cls.requires_grad
        s = (
            f"num_channels={self.num_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            f"normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        return s

class Classifier(nn.Module):
    def __init__(self, classifier_type='cosine', num_channels=512, num_classes=4, bias=False):
        super().__init__()

        self.classifier_type = classifier_type
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.global_pooling = False

        if self.classifier_type == "cosine":
            self.layers = CosineClassifier(
                num_channels=self.num_channels,
                num_classes=self.num_classes,
                scale=10.0,
                learn_scale=True,
                bias=bias,
            )

        elif self.classifier_type == "linear":
            self.layers = nn.Linear(self.num_channels, self.num_classes, bias=bias)
            if bias:
                self.layers.bias.data.zero_()

            fout = self.layers.out_features
            self.layers.weight.data.normal_(0.0, np.sqrt(2.0 / fout))

        elif self.classifier_type == "mlp_linear":
            mlp_channels = [int(num_channels / 2), int(num_channels / 4)]
            num_mlp_channels = len(mlp_channels)
            mlp_channels = [self.num_channels,] + mlp_channels
            self.layers = nn.Sequential()

            pre_act_relu = False
            if pre_act_relu:
                self.layers.add_module("pre_act_relu", nn.ReLU(inplace=False))

            for i in range(num_mlp_channels):
                self.layers.add_module(
                    f"fc_{i}", nn.Linear(mlp_channels[i], mlp_channels[i + 1], bias=False),
                )
                self.layers.add_module(f"bn_{i}", nn.BatchNorm1d(mlp_channels[i + 1]))
                self.layers.add_module(f"relu_{i}", nn.ReLU(inplace=True))

            fc_prediction = nn.Linear(mlp_channels[-1], self.num_classes)
            fc_prediction.bias.data.zero_()
            self.layers.add_module("fc_prediction", fc_prediction)
        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(self.classifier_type)
            )

    def flatten(self):
        return (
            self.classifier_type == "linear"
            or self.classifier_type == "cosine"
            or self.classifier_type == "mlp_linear"
        )

    def forward(self, features):
        if self.global_pooling:
            features = global_pooling(features, pool_type="avg")

        if features.dim() > 2 and self.flatten():
            features = features.view(features.size(0), -1)

        scores = self.layers(features)
        return scores

def classifier(classifier_type='cosine', num_channels=512, num_classes=4, bias=False):
    classifier = Classifier(
        classifier_type = classifier_type,
        num_channels = num_channels,
        num_classes = num_classes,
        bias = bias
    )
    return classifier

class ConvnetPlusClassifier(nn.Module):
    def __init__(self, num_classes, block, inplanes, planes, stride=2, downsample=None):
        super().__init__()

        self.layers = nn.Sequential(
            block(inplanes, planes, stride=stride, downsample=nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))),
            block(planes * block.expansion, planes, stride=stride, downsample=nn.Sequential(
                nn.Conv2d(planes * block.expansion, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))),
            Classifier(classifier_type='cosine', num_channels=planes * block.expansion, num_classes=num_classes, bias=False)
        )
        # self.layers = nn.Sequential(
        #     nn.Conv2d(inplanes, inplanes * 2, kernel_size=3, padding=1, stride=2, bias=False),
        #     nn.Conv2d(inplanes * 2, inplanes * 4, kernel_size=3, padding=1, stride=2, bias=False),
        #     Classifier(classifier_type='cosine', num_channels=inplanes * 4, num_classes=num_classes, bias=False)
        # )

    def forward(self, features):
        classification_scores = self.layers(features)
        return classification_scores

def test():
    c = classifier('cosine',512,4)
    x = torch.randn(28,512)
    x1 = c(x)
    print(x1.shape)
# test()