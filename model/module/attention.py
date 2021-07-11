###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch

__all__ = ['PAM_Module', 'CAM_Module', 'DANetHead', 'CBAM', 'DCAM']



class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class NLPM(nn.Module):
    '''
    non local pyramid attention module
    '''

    def __init__(self, in_channels, out_channels, key_channels, psp_size=(1, 4, 8, 16), bn=False, res=True, v_conv_kernel=1):

        super(NLPM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.query_channels = key_channels
        self.psp_size = psp_size
        self.bn = bn
        self.res = res

        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False)

        self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False)
        self.key_psp = PSPModule(sizes=psp_size)

        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=v_conv_kernel, padding=v_conv_kernel//2, bias=False)
        self.value_psp = PSPModule(sizes=psp_size)


        if bn:
            self.query_bn = nn.BatchNorm2d(key_channels)
            self.key_bn = nn.BatchNorm2d(key_channels)

        print('Initing NLPM')

        self.softmax = Softmax(dim=-1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        B, C, H, W = x.size()

        query= self.query_conv(x)
        if self.bn:
            query = self.query_bn(query)
        query = query.view(B, self.key_channels, H*W).permute(0, 2, 1)

        key = self.key_conv(x)
        if self.bn:
            key = self.key_bn(key)
        key = self.key_psp(key)

        value = self.value_conv(x)
        value = self.value_psp(value).permute(0, 2, 1)

        atten = torch.bmm(query, key)
        atten = self.softmax(atten)

        new_value = torch.bmm(atten, value)
        new_value = new_value.permute(0, 2, 1).view(B, self.out_channels, H, W)

        if self.res:
            #out = self.residual_conv(x)
            #alpha = torch.sigmoid(self.alpha)
            #out = alpha * new_value + (1-alpha)*out 
            out = x + new_value
        else:
            out = new_value
        
        return out

class Identify(nn.Module):

    def __init__(self):
        super(Identify, self).__init__()

    def forward(self, x):
        return x


class PAPAModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 4, 8), dimension=2):
        super(PAPAModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        #size = _pair(size)
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats, atten):
        n, c, _, _ = feats.size()
        priors = []
        for indx, stage in enumerate(self.stages):
            if atten is not None:
                prior = stage(feats * atten[:,indx:indx+1,:,:]).view(n, c, -1)
            else:
                prior = stage(feats).view(n, c, -1)
            priors.append(prior)

        center = torch.cat(priors, -1)
        return center


class ANAB(nn.Module):
    '''
    ANAB attention module
    '''

    def __init__(self, ch, num_psp, psp_size=[1, 4, 8, 16], with_atten=True):
        super(ANAB, self).__init__()

        print('Init ANAB attention module')

        self.inch = ch
        self.outch = ch

        self.key_num = 0
        for i in psp_size:
            self.key_num += i**2
        self.key_ch = self.key_num // 2
        self.with_atten = with_atten

        self.value_conv   = nn.Conv2d(self.inch, self.outch,   kernel_size=1, bias=False)
        if with_atten:
            self.spatial_conv = nn.Conv2d(self.inch, len(psp_size), kernel_size=1, bias=False)
        self.key_conv     = nn.Conv2d(self.inch, self.key_ch,  kernel_size=1, bias=False)
        self.query_conv   = nn.Conv2d(self.inch, self.key_ch,  kernel_size=1, bias=False)

        self.key_papa = PAPAModule(sizes=psp_size)
        self.value_papa = PAPAModule(sizes=psp_size)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.ind = 1
                
    def forward(self, x):

        B, C, H, W = x.size()

        query= self.query_conv(x)
        #if self.bn:
        #    query = self.query_bn(query)
        query = query.view(B, self.key_ch, H*W).permute(0, 2, 1)
        if self.with_atten:
            psp_atten = self.spatial_conv(x)
            psp_atten = self.sigmoid(psp_atten)
        else:
            psp_atten = None

        #psp_atten = self.spatial_bn(psp_atten)

        key = self.key_conv(x)
        #if self.bn:
        #    key = self.key_bn(key)
        key = self.key_papa(key, psp_atten)

        value = self.value_conv(x)
        value = self.value_papa(value, psp_atten).permute(0, 2, 1)

        atten = torch.bmm(query, key)
        atten = self.softmax(atten)


        new_value = torch.bmm(atten, value)
        new_value = new_value.permute(0, 2, 1).view(B, self.outch, H, W)

        out = new_value + x

        return out.contiguous()

