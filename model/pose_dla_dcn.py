from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .DCNv2.dcn_v2 import DCN
#from .DCNv2.modules.deform_conv import DeformConvPack as DCN
#from .DCNv2.deform_conv_v2 import DeformConv2d as DCN
#from .DCNs_package.pytorch_deform_conv.torch_deform_conv.layers import ConvOffset2D
from lib.core import load_weights
from lib.rpn_util import save_tensor

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=3, padding=1, dilation=1):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv= nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1, groups=num_rows)

    def forward(self, x):

        b, c, h, w = x.size()

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        #print('-------------------------')
        #print('x.size = ', x.size)
        #print('t = ', t)
        #print('##########################')
        #if t < 1:
        #    print("t<1")
        #    return self.conv2d(x)
        #print("t>1")
        x_0 = x.unfold(2, t + self.pad*2, t)
        x_0 = x_0.permute([0, 2, 1, 4, 3]).contiguous()
        x_0 = x_0.view(b, c * self.num_rows, t + self.pad*2, (w+self.pad*2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x_0)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)
        #if t/2 > 1: 
        #    x_1 = x[:, :, int(t/2):-int(t/2), :]
        #    #print("x1.size() = ", x_1.size())
        #    #print("x.size() = ", x.size())
        #    #print("t:", t)
        #    x_1 = x_1.unfold(2, t + self.pad*2, t)
        #    x_1 = x_1.permute([0, 2, 1, 4, 3]).contiguous()
        #    x_1 = x_1.view(b, c * (self.num_rows-1), t + self.pad*2, (w+self.pad*2)).contiguous()

        #    # group convolution for efficient parallel processing
        #    y_1 = self.group_conv_1(x_1)
        #    y_1 = y_1.view(b, self.num_rows-1, self.out_channels, t, w).contiguous()
        #    y_1 = y_1.permute([0, 2, 1, 3, 4]).contiguous()
        #    y_1 = y_1.view(b, self.out_channels, h-t, w)

        #    y[:, :, int(t/2):-int(t/2), :] = 0.5 * y[:, :, int(t/2):-int(t/2), :] + 0.5 * y_1

        return y

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1,
                               bias=True, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1,
                               bias=True, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class DepthBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, num_rows=16):
        super(DepthBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                       stride=1, padding=1,
        #                       bias=False, dilation=dilation)
        self.conv_depth = LocalConv2d(num_rows, planes, planes, kernel=3, padding=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        #self.group_conv = nn.Conv2d(planes * num_rows, planes * num_rows, 3, stride=1, groups=num_rows, padding=1)
        #self.group_conv_1 = nn.Conv2d(planes * (num_rows-1), planes * (num_rows-1), 3, stride=1, groups=num_rows-1,padding=1)

        self.stride = stride
        self.num_rows = num_rows
        self.out_channels = planes

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #-----depth aware conv
        out = self.conv_depth(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        #self.conv2 = DCN(bottle_planes, bottle_planes, kernel_size=3,
        #                       stride=stride, padding=dilation,
        #                       bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)

            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            #self.downsample = nn.MaxPool2d(stride, stride=stride, padding=(root_kernel_size- 1) // 2)
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.LeakyReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)

        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)

        try:
            self.load_state_dict(model_weights)
        except:
            load_weights(self, path=None, src_weights=model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def dla34_depth(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=DepthBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

def dla102(pretrained=None, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained is not None:
        model.load_pretrained_model(data='imagenet', name='dla102', hash='d94d9790')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1)
        #self.conv = DCN(chi, cho, kernel_size=3, stride=1, padding=1)
        print('init deform conv...')

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class DeformLocConv(nn.Module):
    def __init__(self, chi, cho, num_rows):
        super(DeformLocConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.LeakyReLU(inplace=True)
        )
        self.num_rows = num_rows
        self.conv = DCN(chi*num_rows, cho*num_rows, kernel_size=(3,3), stride=1, padding=0, dilation=1, deformable_groups=num_rows)
        self.pad = 1
        self.out_channels = cho

    def forward(self, x):

        b, c, h, w = x.size()
        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)
        t = int(h / self.num_rows)

        x = x.unfold(2, t + self.pad*2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad*2, (w+self.pad*2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w).contiguous()
        
        #x = self.conv(x)
        y = self.actf(y)
        return y

class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f, conf):
        super(IDAUp, self).__init__()
        self.out_channels = channels

        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  

            if conf.ida_dcnv2:
                proj = DeformConv(c, o)
                node = DeformConv(o, o)
            else:
                proj = nn.Conv2d(c, o, 3, 1, 1)
                node = nn.Conv2d(o, o, 3, 1, 1)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, conf=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:], scales[j:]//scales[j], conf=conf))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x

class NL_Up(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, q_ch, v_ch):
        super(NL_Up, self).__init__()

        #self.query_dim = q_ch//8
        if v_ch != q_ch:
            self.v_conv = nn.Conv2d(v_ch, q_ch, 1, bias=False)
            self.k_conv = nn.Conv2d(v_ch, q_ch, 1, bias=False)
        else:
            self.v_conv = Identity()
            self.k_conv = Identity()

        self.q_bn = nn.BatchNorm2d(q_ch, momentum=BN_MOMENTUM)
        self.k_bn = nn.BatchNorm2d(q_ch, momentum=BN_MOMENTUM)
        self.sml_ch = q_ch


        self.softmax = nn.Softmax(dim=-1)
        self.ind = 0
    def forward(self, q, v):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        B, q_ch, q_h, q_w = q.size()
        B, v_ch, v_h, v_w = v.size()

        q = self.q_bn(q)
        query = q.view(B, self.sml_ch, q_h*q_w).permute(0, 2, 1)

        key = self.k_conv(v)
        key = self.k_bn(key).view(B, self.sml_ch, v_h*v_w)

        value = self.v_conv(v).view(B, self.sml_ch, v_h*v_w).permute(0, 2, 1)

        sim = torch.bmm(query, key)
        attention = self.softmax(sim)

        #save_tensor(attention, './{}.npy'.format(self.ind))
        #self.ind += 1
        out = torch.bmm(attention, value).permute(0, 2, 1)
        out = out.view(B, q_ch, q_h, q_w)

        return out

class DLASeg(nn.Module):
    def __init__(self, base_name, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, conf, out_channel=0):
        super(DLASeg, self).__init__()
        self.features = []
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales, conf=conf)

        if out_channel == 0:
            out_channel = channels[self.first_level]
            self.out_channels = out_channel

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)], conf)

        
        #self.heads = heads
        #for head in self.heads:
        #    classes = self.heads[head]
        #    if head_conv > 0:
        #      fc = nn.Sequential(
        #          nn.Conv2d(channels[self.first_level], head_conv,
        #            kernel_size=3, padding=1, bias=True),
        #          nn.LeakyReLU(inplace=True),
        #          nn.Conv2d(head_conv, classes, 
        #            kernel_size=final_kernel, stride=1, 
        #            padding=final_kernel // 2, bias=True))
        #      if 'hm' in head:
        #        fc[-1].bias.data.fill_(-2.19)
        #      else:
        #        fill_fc_weights(fc)
        #    else:
        #      fc = nn.Conv2d(channels[self.first_level], classes, 
        #          kernel_size=final_kernel, stride=1, 
        #          padding=final_kernel // 2, bias=True)
        #      if 'hm' in head:
        #        fc.bias.data.fill_(-2.19)
        #      else:
        #        fill_fc_weights(fc)
        #    self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return y[-1]

        #z = {}
        #for head in self.heads:
        #    z[head] = self.__getattr__(head)(y[-1])
        #return [z]
    

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
#def build(conf, phase='train'):
  model = DLASeg('dla{}'.format(num_layers), 
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model

