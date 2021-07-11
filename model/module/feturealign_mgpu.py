import torch.nn as nn
import torch
from model.DCNv2.dcn_v2 import DCNv2
from torch.nn.modules.utils import _pair
import numpy as np

class center_align(nn.Module):

    def __init__(self, ch, anchors, xy_mean, xy_std, feat_stride, feat_size, kernel_size=1, k=1, thresh=0.5):
        super(center_align, self).__init__()
        print('Initing center align')
        dev = torch.tensor([0], dtype=torch.float, requires_grad=False)
        self.device = dev.device
        self.ch = ch
        self.kernel_size = _pair(kernel_size)
        #self.anchors = torch.tensor(anchors, dtype=torch.float, requires_grad=False, device=self.device)
        self.anchors = anchors.clone().float().detach().to(self.device)
        #self.anchors = anchors
        self.num_anchors = anchors.shape[0]
        self.k = k
        self.feat_stride = feat_stride
        self.thresh = thresh
        self.xy_mean = torch.tensor(xy_mean, dtype=torch.float, requires_grad=False, device=self.device) 
        self.xy_std = torch.tensor(xy_std, dtype=torch.float, requires_grad=False, device=self.device)
        self.feat_size = feat_size

        anchors_w = (self.anchors[:, 2] - self.anchors[:, 0]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        anchors_h = (self.anchors[:, 3] - self.anchors[:, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.anchors_w = anchors_w / self.feat_stride
        self.anchors_h = anchors_h / self.feat_stride

        self.align = DCNv2(ch, ch, self.kernel_size, 1, kernel_size//2, dilation=1, deformable_groups=1)

        #self.proj_actf = nn.Sequential(
        #	         nn.Conv2d(ch * 2, ch, kernel_size=1),
        #                 nn.BatchNorm2d(ch),
        #                 nn.LeakyReLU()
        #                  )

        #self.proj = nn.Conv2d(ch*2, ch, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        #self.mask = torch.ones([1, self.kernel_size[0]*self.kernel_size[1], self.feat_size[0], self.feat_size[1]],
        #                            dtype=torch.float, requires_grad=False)
        self.indx = -1

    def forward(self, x, bbox_x, bbox_y, prob):

        batch_size, c, feat_h, feat_w = x.size()

        if self.xy_std.device != x.device:
            self.xy_std = self.xy_std.to(x.device)
            self.xy_mean = self.xy_mean.to(x.device)
            self.anchors_w = self.anchors_w.to(x.device)
            self.anchors_h = self.anchors_h.to(x.device)

        prob_k, ind = torch.topk(prob, k=self.k, dim=1)
        softmax_k = self.softmax(prob_k)
        #mask = torch.sum(prob_k * softmax_k, dim=1, keepdim=True)
        mask, _ = torch.max(prob_k, dim=1, keepdim=True)
        hard_mask = (mask>self.thresh).float()
        #print('bbox_x.device:', bbox_x.device)
        #print('self.xy_std.device:', self.xy_std.device)
        #print('self.xy_mean.device:', self.xy_mean.device)
        #print('self.device:', self.device)
        offset_x = (bbox_x * self.xy_std[0] + self.xy_mean[0]) * self.anchors_w 
        offset_y = (bbox_y * self.xy_std[1] + self.xy_mean[1]) * self.anchors_h 

        offset_x = torch.gather(offset_x, dim=1, index=ind)
        offset_x = torch.sum(offset_x*softmax_k, dim=1, keepdim=True) * hard_mask
        offset_y = torch.gather(offset_y, dim=1, index=ind)
        offset_y = torch.sum(offset_y*softmax_k, dim=1, keepdim=True) * hard_mask


        offset = torch.cat([offset_y, offset_x], dim=1)
        offset = offset.repeat(1, self.kernel_size[0]*self.kernel_size[1], 1, 1)

        #print('offset.shape: ', offset.shape)
        debug = False
        path = './output/debug/'
        if debug:
            self.indx += 1
            np.save(path+'ct_prob_{}.npy'.format(self.indx), prob.cpu().numpy())
            np.save(path+'ct_mask_{}.npy'.format(self.indx), mask.cpu().numpy())
            np.save(path+'ct_hard_mask_{}.npy'.format(self.indx), hard_mask.cpu().numpy())
            np.save(path+'ct_offset_{}.npy'.format(self.indx), offset.cpu().numpy())

        mask = mask.repeat(1, self.kernel_size[0]*self.kernel_size[1], 1, 1)
        #mask = hard_mask.repeat(1, self.kernel_size[0]*self.kernel_size[1], 1, 1)
        #mask = self.mask.repeat(batch_size, 1, 1, 1)

        feats = self.align(x, offset, mask)
        #print(feats.shape)
        #out = torch.cat([feats, x], dim=1)
        #out = self.conv(feats)
        #return self.proj_actf(out)
        #return self.proj(out)
        return feats + x


class shape_align(nn.Module):

    def __init__(self, ch, anchors, feat_stride, feat_size, kernel_size=3, k=1, thresh=0.5):
        super(shape_align, self).__init__()
        
        print('Initing shape align')

        self.ch = ch
        self.kernel_size = _pair(kernel_size)
        #self.anchors = torch.tensor(anchors, dtype=torch.float, requires_grad=False)
        self.anchors = anchors
        self.num_anchors = anchors.shape[0]
        self.feat_stride = feat_stride
        self.feat_size = feat_size
        self.k = k
        self.thresh = thresh

        anchors_w = (self.anchors[:, 2] - self.anchors[:, 0]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        anchors_h = (self.anchors[:, 3] - self.anchors[:, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        self.offset_h_step = (anchors_h / self.feat_stride / self.kernel_size[0]).repeat(1, 1, int(self.feat_size[0]), int(self.feat_size[1]))
        self.offset_w_step = (anchors_w / self.feat_stride / self.kernel_size[1]).repeat(1, 1, int(self.feat_size[0]), int(self.feat_size[1]))


        offset = torch.zeros((1, self.num_anchors, 2*self.kernel_size[0]*self.kernel_size[1], 
                                int(self.feat_size[0]), int(self.feat_size[1])), dtype=torch.float, requires_grad=False)
        for i in range(self.kernel_size[0]): # h
            for j in range(self.kernel_size[1]): #w
                ind_base = i * self.kernel_size[1] + j
                # h offset
                offset[:, :, ind_base*2, :, :] = (self.offset_h_step-1) * (i - self.kernel_size[0]/2 +0.5)
                # w offset
                offset[:, :, ind_base*2+1, :, :] = (self.offset_w_step-1) * (j - self.kernel_size[1]/2 +0.5)

        self.offset = offset
        self.align = DCNv2(ch, ch, self.kernel_size, 1, kernel_size//2, 1, deformable_groups=1)

        # self.proj_actf = nn.Sequential(
        # 	         nn.Conv2d(ch * 2, ch, kernel_size=1),
        #                  nn.BatchNorm2d(ch),
        #                  nn.LeakyReLU()
        #                   )
        # self.conv = nn.Conv2d(ch, ch, self.kernel_size, 1, kernel_size//2)
        self.proj = nn.Conv2d(ch*2, ch, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        #self.mask = torch.ones([1, self.kernel_size[0]*self.kernel_size[1], self.feat_size[0], self.feat_size[1]],
        #                            dtype=torch.float, requires_grad=False)
        self.indx=-1

    def forward(self, x, prob):

        batch_size, c, feat_h, feat_w = x.size()

        if self.offset.device != x.device:
            self.offset = self.offset.to(x.device)

        prob_k, ind_k = torch.topk(prob, k=self.k, dim=1)
        softmax_k = self.softmax(prob_k)
        #mask = torch.sum(prob_k * softmax_k, dim=1, keepdim=True)
        mask, _ = torch.max(prob_k, dim=1, keepdim=True)
        hard_mask = (mask>self.thresh).float()

        offset = self.offset.repeat(batch_size, 1, 1, 1, 1)

        ind_k_expand = ind_k.unsqueeze(2).expand(batch_size, -1, self.kernel_size[0]*self.kernel_size[1]*2, -1, -1)
        offset = torch.gather(offset, dim=1, index=ind_k_expand)
        offset = torch.sum(offset*(softmax_k.unsqueeze(2)), dim=1)
        
        offset = offset * hard_mask

        debug = False
        path = './output/debug/'
        if debug:
            self.indx += 1
            np.save(path+'shape_prob_{}.npy'.format(self.indx), prob.cpu().numpy())
            np.save(path+'shape_mask_{}.npy'.format(self.indx), mask.cpu().numpy())
            np.save(path+'shape_hard_mask_{}.npy'.format(self.indx), hard_mask.cpu().numpy())
            np.save(path+'shape_offset_{}.npy'.format(self.indx), offset.cpu().numpy())
        #mask = self.mask.repeat(batch_size, 1, 1, 1)
        mask = mask.repeat(1, self.kernel_size[0]*self.kernel_size[1], 1, 1)
        #mask = hard_mask.repeat(1, self.kernel_size[0]*self.kernel_size[1], 1, 1)

        # offset_x3d = (bbox_x3d.detach() * self.bbox_stds[4] + self.bbox_means[4]) * anchors_w / self.feat_stride
        # offset_y3d = (bbox_y3d.detach() * self.bbox_stds[5] + self.bbox_means[5]) * anchors_h / self.feat_stride

        # if debug:
        #     np.save(path+'offset_x3d_{}.npy'.format(self.indx), offset_x3d.cpu().numpy())
        #     np.save(path+'offset_y3d_{}.npy'.format(self.indx), offset_y3d.cpu().numpy())

        # if self.align_type == 'mean':
        #     offset_x3d = torch.mean(offset_x3d * weights, dim = 1, keepdim=True)
        #     offset_y3d = torch.mean(offset_y3d * weights, dim = 1, keepdim=True)
        # elif self.align_type == 'max':
        #     offset_x3d = offset_x3d[self.ind_0, ind_1, self.ind_2, self.ind_3].unsqueeze(1)*mask
        #     offset_32d = offset_y3d[self.ind_0, ind_1, self.ind_2, self.ind_3].unsqueeze(1)*mask

        # if debug:
        #     np.save(path+'offset_x3d_mean_{}.npy'.format(self.indx), offset_x3d.cpu().numpy())
        #     np.save(path+'offset_y3d_mean_{}.npy'.format(self.indx), offset_y3d.cpu().numpy())
        feats_aligned = self.align(x, offset, mask)
        #feats = self.conv(x)
        #out = torch.cat([feats_aligned, x], dim=1)
        #return self.proj(out)
        #return self.proj_actf(out)
        return feats_aligned + x


