import torch.nn as nn
from torchvision import models
from lib.rpn_util import *
import torch
from model.pose_dla_dcn import DLASeg, DeformConv
from model.DCNv2.dcn_v2 import DCNv2
from model.module.attention import  ANAB
from torch.nn.modules.utils import _pair
from model.module.feturealign_mgpu import shape_align, center_align


def dilate_layer(layer, val):

    layer.dilation = val
    layer.padding = val

class Shift_Module(nn.Module):
    def __init__(self, inch, outch):
        super(Shift_Module, self).__init__()

        self.inch = inch
        self.outch = outch

        self.shift = DeformConv(inch, outch)

    def forward(self, x):
        shift_x = self.shift(x)

        return x+shift_x

class RPN(nn.Module):


    def __init__(self, phase, base, conf):
        super(RPN, self).__init__()

        self.base = base

        # settings
        self.phase = phase
        self.device = conf.device
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]
        self.anchors = torch.tensor(conf.anchors, dtype=torch.float, device=self.device, requires_grad=False)
        self.bbox_means = conf.bbox_means[0]
        self.bbox_stds = conf.bbox_stds[0]
        self.base_channels = self.base.out_channels
        self.head_channels = 256
        self.back_bone = conf.back_bone
        self.batch_size = conf.batch_size
        if 'align_type' in conf:
            self.align_type = conf.align_type
        else:
            self.align_type = 'max'

        if 'attention' in conf:
            self.attention = conf.attention
        else:
            self.attention = None

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.float().to(self.device)

        self.cls = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 3, padding=1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors * self.num_classes, 1),
                    )

        # bbox 2d
        self.bbox_x = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        self.bbox_y = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        self.bbox_w = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        self.bbox_h = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        # bbox 3d
        self.bbox_x3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )
        self.bbox_y3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        # feature align
        if conf.center_align:
            self.center_align2d = center_align(self.base_channels, 
                                  self.anchors, xy_mean=self.bbox_means[0:2], xy_std=self.bbox_stds[0:2],
                                  feat_stride=self.feat_stride, feat_size=self.feat_size, kernel_size=1, k=1, thresh=0.5)

            self.center_align3d = center_align(self.base_channels, 
                                  self.anchors, xy_mean=self.bbox_means[4:6], xy_std=self.bbox_stds[4:6],
                                  feat_stride=self.feat_stride, feat_size=self.feat_size, kernel_size=1, k=1, thresh=0.5)

        else:
            self.center_align2d = None
            self.center_align3d = None

        if conf.shape_align:
            self.shape_align = shape_align(self.base_channels, self.anchors, 
                                feat_stride=self.feat_stride, feat_size=self.feat_size,
                                kernel_size=3, k=1, thresh=0.5)
        else:
            self.shape_align = None


        self.bbox_z3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels , self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )
        if self.attention == "ANAB":
            self.bbox_z3d_gl = nn.Sequential(
                            ANAB(self.base_channels, 1),
                            nn.BatchNorm2d(self.base_channels),
                            nn.LeakyReLU(inplace=True),
                       )

        self.bbox_w3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )
        self.bbox_h3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )
        self.bbox_l3d  = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )
        self.bbox_rY3d = nn.Sequential(
                            nn.Conv2d(self.base_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.head_channels, 1),
                            nn.BatchNorm2d(self.head_channels),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(self.head_channels, self.num_anchors, 1),
                        )

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        batch_size = x.size(0)

        x = self.base(x)

        assert x.shape[2] == self.feat_size[0], 'x.shape is {}'.format(x.shape)

        cls = self.cls(x)

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        fg_prob = (1 - prob.detach()[:, 0, :, :]).view(batch_size, self.num_anchors, feat_h, feat_w)
        
        if self.shape_align:
            feats = self.shape_align(x, fg_prob)
        else:
            feats = x
        # bbox 2d
        bbox_x = self.bbox_x(feats)

        bbox_y = self.bbox_y(feats)

        if self.center_align2d:
            feats_align2d = self.center_align2d(feats, bbox_x.detach(), bbox_y.detach(), fg_prob)
        else:
            feats_align2d = feats

        bbox_w = self.bbox_w(feats_align2d)

        bbox_h = self.bbox_h(feats_align2d)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(feats)

        bbox_y3d = self.bbox_y3d(feats)

        if self.center_align3d:
            feats_align3d = self.center_align3d(feats, bbox_x3d.detach(), bbox_y3d.detach(), fg_prob)
        else:
            feats_align3d = feats

        bbox_w3d = self.bbox_w3d(feats_align3d)

        bbox_h3d = self.bbox_h3d(feats_align3d)

        bbox_l3d = self.bbox_l3d(feats_align3d)

        bbox_rY3d = self.bbox_rY3d(feats_align3d)

        feats_z = feats_align3d
        if self.attention == "ANAB":
            feats_gl = self.bbox_z3d_gl(feats_z)
        else:
            feats_gl = feats_z
        bbox_z3d = self.bbox_z3d(feats_gl)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        #feat_size = [feat_h, feat_w]
        feat_size = torch.tensor([feat_h, feat_w], dtype=torch.float, device=self.device)

        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'
    base_name = conf.back_bone
    if base_name[0:3] =='dla':
        base = DLASeg(base_name, pretrained=conf.pre_train, down_ratio=conf.feat_stride, 
                        final_kernel=1, last_level=5, head_conv=256, conf=conf)
        rpn_net = RPN(phase, base, conf)
    else:
        raise NotImplementedError


    if train: rpn_net.train()
    else: rpn_net.eval()

    return rpn_net
