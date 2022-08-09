###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn

import logging
import math

logger = logging.getLogger(__name__)

torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'get_attention_module_instance']


class DANetHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_layer: nn.Module,
                 module_class: type,
                 dim_collapsion: int=2):
        super(DANetHead, self).__init__()

        inter_channels = in_channels // dim_collapsion

        self.conv5c = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.attention_module = module_class(inter_channels)
        self.conv52 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):

        feat2 = self.conv5c(x)
        sc_feat = self.attention_module(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        return sc_output

class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        # proj_query = proj_query / torch.norm(proj_query, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        # proj_key = proj_key / torch.norm(proj_key, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out

class Normed_PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(Normed_PAM_Module, self).__init__()
        self.channel_in = in_dim
        # print(in_dim)
        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        proj_query = proj_query / torch.norm(proj_query, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        proj_key = proj_key / torch.norm(proj_key, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        # proj_query = proj_query / torch.norm(proj_query, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        # proj_key = proj_key / torch.norm(proj_key, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        energy = torch.bmm(proj_query, proj_key)
        # print("Energy: ",energy.shape)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        # print("Max_energy_0: ",max_energy_0.shape)
        energy_new = max_energy_0 - energy
        # print("Energy: ",energy_new.shape )
        attention = self.softmax(energy_new)
        # print("Attention: ",attention.shape)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        # print("Out: ",out.shape)
        logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out

class Normed_CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(Normed_CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        proj_query = proj_query / torch.norm(proj_query, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        proj_key = proj_key / torch.norm(proj_key, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out

class Second_Order_CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        N = height * width
        ave_I = 1.0 / N * (torch.eye(N, device=x.device) - 1.0 / N * torch.ones(N, device=x.device))
        ave_I = ave_I.unsqueeze(0).repeat(m_batchsize, 1, 1)

        sigma = torch.bmm(proj_query, ave_I)
        sigma = torch.bmm(sigma, proj_key)
        sigma = sigma / math.sqrt(N)

        attention = self.softmax(sigma)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out

class Second_Order_PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(Second_Order_PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        C_ = proj_query.shape[-1]
        ave_I = 1.0 / C_ * (torch.eye(C_, device=x.device) - 1.0 / C_ * torch.ones(C_, device=x.device))
        ave_I = ave_I.unsqueeze(0).repeat(m_batchsize, 1, 1)

        sigma = torch.bmm(proj_query, ave_I)
        sigma = torch.bmm(sigma, proj_key)
        sigma = sigma / math.sqrt(C_)

        attention = self.softmax(sigma)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out

class Multi_Order_CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.beta = Parameter(0.5 * torch.ones(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        N = height * width
        ave_I = 1.0 / N * (torch.eye(N, device=x.device) - 1.0 / N * torch.ones(N, device=x.device))
        ave_I = ave_I.unsqueeze(0).repeat(m_batchsize, 1, 1)

        sigma = torch.bmm(proj_query, ave_I)
        sigma = torch.bmm(sigma, proj_key)
        sigma = sigma / math.sqrt(N)

        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy

        attention = self.beta * self.softmax(sigma) + (1.0 - self.beta) * self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        logging.debug('cam device: {}, {}'.format(out.device, self.gamma.device))
        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out

class Multi_Order_PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(Multi_Order_PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.key_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1
        )
        self.value_conv = Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1
        )
        self.gamma = Parameter(torch.zeros(1))
        self.beta = Parameter(0.5 * torch.ones(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        C_ = proj_query.shape[-1]
        ave_I = 1.0 / C_ * (torch.eye(C_, device=x.device) - 1.0 / C_ * torch.ones(C_, device=x.device))
        ave_I = ave_I.unsqueeze(0).repeat(m_batchsize, 1, 1)

        sigma = torch.bmm(proj_query, ave_I)
        sigma = torch.bmm(sigma, proj_key)
        sigma = sigma / math.sqrt(C_)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.beta * self.softmax(sigma) + (1 - self.beta) * self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(
            proj_value,
            attention.permute(0, 2, 1)
        )
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out

def test():
    # Attention = Normed_PAM_Module().forward(x)
    x = torch.randn(5,512,28,28)
    y = CAM_Module(Module).forward(x)
    print(y.shape)
# test()
