import torch.nn as nn
import torch.nn.functional as F
import sys

# stop python from writing so much bytecode
#sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.rpn_util import *


class RPN_3D_loss(nn.Module):

    def __init__(self, conf):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h
        self.device = conf.device


    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size):

        stats = []
        #loss = torch.tensor(0).type(torch.cuda.FloatTensor)
        loss = torch.tensor(0, dtype=torch.float, device=self.device)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        batch_size = cls.shape[0]

        prob_detach = prob.cpu().detach().numpy()

        bbox_x = bbox_2d[:, :, 0]
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        bbox_x3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_y3d_proj = torch.zeros(bbox_x3d.shape)
        bbox_z3d_proj = torch.zeros(bbox_x3d.shape)

        labels = np.zeros(cls.shape[0:2], dtype=int)
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        ious_2d = torch.zeros(cls.shape[0:2])
        ious_3d = torch.zeros(cls.shape[0:2])
        coords_abs_z = torch.zeros(cls.shape[0:2])
        coords_abs_ry = torch.zeros(cls.shape[0:2])

        # get all rois
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
        #rois = rois.type(torch.cuda.FloatTensor)
        rois = rois.float().to(self.device)

        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        #src_anchors = self.anchors[rois[:, 4].type(torch.cuda.LongTensor), :]
        src_anchors = self.anchors[rois[:, 4].cpu().int(), :]
        #src_anchors = torch.tensor(src_anchors, requires_grad=False).type(torch.cuda.FloatTensor)
        src_anchors = torch.tensor(src_anchors, requires_grad=False, dtype=torch.float, device=self.device)
        if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        bbox_x3d_dn = bbox_x3d_dn * widths.unsqueeze(0) + ctr_x.unsqueeze(0)
        bbox_y3d_dn = bbox_y3d_dn * heights.unsqueeze(0) + ctr_y.unsqueeze(0)
        bbox_z3d_dn = src_anchors[:, 4].unsqueeze(0) + bbox_z3d_dn
        bbox_w3d_dn = torch.exp(bbox_w3d_dn) * src_anchors[:, 5].unsqueeze(0)
        bbox_h3d_dn = torch.exp(bbox_h3d_dn) * src_anchors[:, 6].unsqueeze(0)
        bbox_l3d_dn = torch.exp(bbox_l3d_dn) * src_anchors[:, 7].unsqueeze(0)
        bbox_ry3d_dn = src_anchors[:, 8].unsqueeze(0) + bbox_ry3d_dn

        for bind in range(0, batch_size):

            imobj = imobjs[bind]
            if type(imobj) is dict:
                imobj = edict(imobj)
            gts = imobj.gts

            #p2_inv = torch.from_numpy(imobj.p2_inv).type(torch.cuda.FloatTensor)
            p2_inv = torch.from_numpy(imobj.p2_inv).float().to(self.device)
            p2 = torch.from_numpy(imobj.p2).float().to(self.device)

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes
            gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                rois = rois.cpu()

                # bbox regression
                transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                  self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                  self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                                  tracker=rois[:, 4].numpy())

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                # normalize 3d
                transforms[:, 5:12] -= self.bbox_means[:, 4:]
                transforms[:, 5:12] /= self.bbox_stds[:, 4:]

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)
                #logging.info('batch[{}]: {}fg_inds, {}bg_inds, {}ign_inds'.format(bind, fg_inds.shape, bg_inds.shape,ign_inds.shape))

                #transforms = torch.from_numpy(transforms).cuda()

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois.shape[0]*self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois.shape[0]*self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:

                    # compile deltas pred
                    deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis ], bbox_y[bind, :, np.newaxis],
                                           bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

                    # compile deltas targets
                    deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                    bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                   axis=1)

                    # move to gpu
                    #deltas_2d_tar = torch.tensor(deltas_2d_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    deltas_2d_tar = torch.tensor(deltas_2d_tar, requires_grad=False, device=self.device, dtype=torch.float)

                    means = self.bbox_means[0, :]
                    stds = self.bbox_stds[0, :]

                    #rois = rois.cuda()
                    rois = rois.to(self.device)

                    coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)
                    coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)
                    #logging.info('deltas_2d_tar[fg_inds]:\n{}'.format(deltas_2d_tar[fg_inds]))
                    #logging.info('rois[fg_inds]:\n{}'.format(rois[fg_inds]))
                    #logging.info('means:\n{}'.format(means))
                    #logging.info('stds:\n{}'.format(stds))

                    ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')
                    #logging.info('ious_2d[bind, fg_inds]:\n{}'.format(ious_2d[bind, fg_inds]))
                    #logging.info('coords_2d[bind, fg_inds]:\n{}'.format(coords_2d[fg_inds]))
                    #logging.info('coords_2d_tar[bind, fg_inds]:\n{}'.format(coords_2d_tar[fg_inds]))

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]

                    #src_anchors = self.anchors[rois[fg_inds, 4].type(torch.cuda.LongTensor), :]
                    src_anchors = self.anchors[rois[fg_inds, 4].cpu().int(), :]
                    #src_anchors = torch.tensor(src_anchors, requires_grad=False).type(torch.cuda.FloatTensor)
                    src_anchors = torch.tensor(src_anchors, requires_grad=False, device=self.device, dtype=torch.float)
                    if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
                    bbox_z3d_dn_fg = bbox_z3d_dn[bind, fg_inds]
                    bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
                    bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
                    bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
                    bbox_ry3d_dn_fg = bbox_ry3d_dn[bind, fg_inds]

                    # re-scale all 2D back to original
                    bbox_x3d_dn_fg /= imobj['scale_factor']
                    bbox_y3d_dn_fg /= imobj['scale_factor']

                    coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_y3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_z3d_dn_fg[np.newaxis,:]), dim=0)
                    coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)

                    coords_3d = torch.mm(p2_inv, coords_2d)
                    #print('coords_3d:', coords_3d)

                    bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :]
                    bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :]
                    bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :]

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                    #bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False, dtype=torch.float, device=self.device)
                    bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                    #bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                    bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False, dtype=torch.float, device=self.device)
                    bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                    coords_abs_z[bind, fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
                    #logging.info('coords_abs_z[bind, fg_inds]:{}'.format(coords_abs_z[bind, fg_inds] ))
                    #logging.info('bbox_z3d_dn_tar:{}'.format(bbox_z3d_dn_tar))
                    #logging.info('bbox_z3d_dn_fg:{}'.format(bbox_z3d_dn_fg))
                    coords_abs_ry[bind, fg_inds] = torch.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg)

            else:

                bg_inds = np.arange(0, rois.shape[0])

                if self.box_samples == np.inf: bg_num = len(bg_inds)
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)


                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC


            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active].astype(int)]
            #logging.info('active:{}'.format(np.flatnonzero(active).shape))

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))
        #logging.info('fg_inds_all:{} \n{}'.format(fg_inds_all.shape, fg_inds_all))
        #logging.info('bg_inds_all:{} \n{}'.format(bg_inds_all.shape, bg_inds_all))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = cls.argmax(dim=2).cpu().detach().numpy()
        #logging.info('cls_pred:')
        #logging.info(cls_pred[fg_inds_unravel])
        #logging.info('labels:')
        #logging.info(labels[fg_inds_unravel])

        #logging.info('bg_pred:')
        #logging.info(cls_pred[bg_inds_unravel])
        #logging.info('bg_labels:')
        #logging.info(labels[bg_inds_unravel])

        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)
        #logging.info('box weighting fg_num:{}, bg_num{}'.format(fg_num, bg_num))

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        #labels = torch.tensor(labels, requires_grad=False)
        #labels = labels.view(-1).type(torch.cuda.LongTensor)
        labels = torch.tensor(labels, requires_grad=False, dtype=torch.long, device=self.device).view(-1)

        #labels_weight = torch.tensor(labels_weight, requires_grad=False)
        #labels_weight = labels_weight.view(-1).type(torch.cuda.FloatTensor)
        labels_weight = torch.tensor(labels_weight, requires_grad=False, dtype=torch.float, device=self.device).view(-1)

        cls = cls.view(-1, cls.shape[2])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0
            #logging.info('cls active:{}'.format(np.flatnonzero(active).shape))
            #logging.info('cls active:{}'.format(np.flatnonzero(active)))

            if np.any(active.cpu().numpy()):

                loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                #logging.info('cls_loss:{}'.format(loss_cls))
                stats.append({'name': 'cls', 'val': loss_cls, 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            #bbox_weights = torch.tensor(bbox_weights, requires_grad=False).type(torch.cuda.FloatTensor).view(-1)
            bbox_weights = torch.tensor(bbox_weights, requires_grad=False, dtype=torch.float, device=self.device).view(-1)

            active = bbox_weights > 0
            #logging.info('bbox active:{}'.format(np.flatnonzero(active).shape))
            #logging.info('bbox active:{}'.format(np.flatnonzero(active)))
            #logging.info('bbox_weights:{}'.format(bbox_weights[active]))

            if self.bbox_2d_lambda:

                # bbox loss 2d
                #bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                #logging.info('bbox_2d_loss:{}'.format(bbox_2d_loss))
                stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})


            if self.bbox_3d_lambda:

                # bbox loss 3d
                #bbox_x3d_tar = torch.tensor(bbox_x3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_y3d_tar = torch.tensor(bbox_y3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_z3d_tar = torch.tensor(bbox_z3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_w3d_tar = torch.tensor(bbox_w3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_h3d_tar = torch.tensor(bbox_h3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_l3d_tar = torch.tensor(bbox_l3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                #bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_x3d_tar  = torch.tensor(bbox_x3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_y3d_tar  = torch.tensor(bbox_y3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_z3d_tar  = torch.tensor(bbox_z3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_w3d_tar  = torch.tensor(bbox_w3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_h3d_tar  = torch.tensor(bbox_h3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_l3d_tar  = torch.tensor(bbox_l3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)
                bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False, dtype=torch.float, device=self.device).view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_ry3d = bbox_ry3d[:, :].view(-1)

                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none')

                loss_bbox_x3d = (loss_bbox_x3d * bbox_weights[active]).mean()
                loss_bbox_y3d = (loss_bbox_y3d * bbox_weights[active]).mean()
                loss_bbox_z3d = (loss_bbox_z3d * bbox_weights[active]).mean()
                loss_bbox_w3d = (loss_bbox_w3d * bbox_weights[active]).mean()
                loss_bbox_h3d = (loss_bbox_h3d * bbox_weights[active]).mean()
                loss_bbox_l3d = (loss_bbox_l3d * bbox_weights[active]).mean()
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active]).mean()

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d + loss_bbox_ry3d)

                bbox_3d_loss *= self.bbox_3d_lambda

                loss += bbox_3d_loss
                #logging.info('bbox_3d_loss:{}'.format(bbox_3d_loss))
                stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_proj_lambda:

                # bbox loss 3d
                bbox_x3d_proj_tar = torch.tensor(bbox_x3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_y3d_proj_tar = torch.tensor(bbox_y3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
                bbox_z3d_proj_tar = torch.tensor(bbox_z3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

                bbox_x3d_proj = bbox_x3d_proj[:, :].view(-1)
                bbox_y3d_proj = bbox_y3d_proj[:, :].view(-1)
                bbox_z3d_proj = bbox_z3d_proj[:, :].view(-1)

                loss_bbox_x3d_proj = F.smooth_l1_loss(bbox_x3d_proj[active], bbox_x3d_proj_tar[active], reduction='none')
                loss_bbox_y3d_proj = F.smooth_l1_loss(bbox_y3d_proj[active], bbox_y3d_proj_tar[active], reduction='none')
                loss_bbox_z3d_proj = F.smooth_l1_loss(bbox_z3d_proj[active], bbox_z3d_proj_tar[active], reduction='none')

                loss_bbox_x3d_proj = (loss_bbox_x3d_proj * bbox_weights[active]).mean()
                loss_bbox_y3d_proj = (loss_bbox_y3d_proj * bbox_weights[active]).mean()
                loss_bbox_z3d_proj = (loss_bbox_z3d_proj * bbox_weights[active]).mean()

                bbox_3d_proj_loss = (loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj)

                bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

                loss += bbox_3d_proj_loss
                stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss, 'format': '{:0.4f}', 'group': 'loss'})

            coords_abs_z = coords_abs_z.view(-1)
            #logging.info('misc_z:{}'.format(coords_abs_z[active].mean()))
            stats.append({'name': 'z', 'val': coords_abs_z[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            coords_abs_ry = coords_abs_ry.view(-1)
            #logging.info('misc_ry:{}'.format(coords_abs_ry[active].mean()))
            stats.append({'name': 'ry', 'val': coords_abs_ry[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            ious_2d = ious_2d.view(-1)
            #logging.info('ious_2d:{}'.format(ious_2d[active].mean()))
            stats.append({'name': 'iou', 'val': ious_2d[active].mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                iou_2d_loss = iou_2d_loss.mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                #logging.info('iou_2d_loss:{}'.format(iou_2d_loss))
                stats.append({'name': 'iou', 'val': iou_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})


        return loss, stats

class RPN_3D_loss_smp(nn.Module):

    def __init__(self, conf):

        super(RPN_3D_loss_smp, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.device = conf.device
        self.num_anchors = conf.anchors.shape[0]
        #self.anchors = conf.anchors.astype(np.float32)
        self.bbox_means = torch.from_numpy(conf.bbox_means).to(self.device).float()
        self.bbox_stds = torch.from_numpy(conf.bbox_stds).to(self.device).float()
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size
        #self.decouple = conf.decouple

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.anchors = torch.from_numpy(conf.anchors).float().to(self.device)
        self.anchors.requires_grad = False



    #def forward(self, cls, prob, bbox_2d, bbox_3d, feat_size, target, meta):
    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size):

        stats = []
        loss = torch.tensor(0, dtype=torch.float, device=self.device)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000


        labels_fg  = imobjs['labels_fg']
        labels_bg  = imobjs['labels_bg']
        labels_ign = imobjs['labels_ign']
        labels     = imobjs['labels'].to(self.device)
        bbox_2d_tar= imobjs['bbox_2d'].to(self.device)
        bbox_3d_tar= imobjs['bbox_3d'].to(self.device)
        batch_rois = imobjs['meta']['rois'].to(self.device)

        labels.requires_grad = False
        bbox_2d_tar.requires_grad = False
        bbox_3d_tar.requires_grad = False
        batch_rois.requires_grad = False


        rois = batch_rois[0]
        any_val = imobjs['meta']['any_val']
        p2 = imobjs['meta']['p2']

        # x, y, z, w, h, l, ry
        bbox_x3d_tar  = bbox_3d_tar[:, :, 0]
        bbox_y3d_tar  = bbox_3d_tar[:, :, 1]
        bbox_z3d_tar  = bbox_3d_tar[:, :, 2]
        bbox_w3d_tar  = bbox_3d_tar[:, :, 3]
        bbox_h3d_tar  = bbox_3d_tar[:, :, 4]
        bbox_l3d_tar  = bbox_3d_tar[:, :, 5]
        bbox_ry3d_tar = bbox_3d_tar[:, :, 6]

        batch_size = cls.shape[0]

        #prob_detach = prob.cpu().detach().numpy()
        prob_detach = prob.detach()

        #bbox_x = bbox_2d[:, :, 0]
        #bbox_y = bbox_2d[:, :, 1]
        #bbox_w = bbox_2d[:, :, 2]
        #bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        #labels = np.zeros(cls.shape[0:2])
        labels_weight = torch.zeros(cls.shape[0:2], device=self.device, requires_grad=False)
        labels_scores = torch.zeros(cls.shape[0:2], device=self.device, requires_grad=False)
        bbox_weights  = torch.zeros(cls.shape[0:2], device=self.device, requires_grad=False)
        batch_fg = torch.zeros(labels_fg.shape, dtype=torch.int, device=self.device)

        #bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        #bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        #bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        ious_2d = torch.zeros(cls.shape[0:2], device=self.device)
        ious_3d = torch.zeros(cls.shape[0:2], device=self.device)

        coords_abs_z  = torch.zeros(cls.shape[0:2], device=self.device)
        coords_abs_ry = torch.zeros(cls.shape[0:2], device=self.device)

        # get all rois
        #rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
        #rois = rois.type(torch.cuda.FloatTensor)

        bbox_x3d_dn  = bbox_x3d * self.bbox_stds[0, 4]  + self.bbox_means[0, 4]
        bbox_y3d_dn  = bbox_y3d * self.bbox_stds[0, 5]  + self.bbox_means[0, 5]
        bbox_z3d_dn  = bbox_z3d * self.bbox_stds[0, 6]  + self.bbox_means[0, 6]
        bbox_w3d_dn  = bbox_w3d * self.bbox_stds[0, 7]  + self.bbox_means[0, 7]
        bbox_h3d_dn  = bbox_h3d * self.bbox_stds[0, 8]  + self.bbox_means[0, 8]
        bbox_l3d_dn  = bbox_l3d * self.bbox_stds[0, 9]  + self.bbox_means[0, 9]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[0, 10] + self.bbox_means[0, 10]
        #bbox_3d_dn  = bbox_3d[:, :, 0:7] * self.bbox_stds[:, 4:11][0]  + self.bbox_means[:, 4:11][0]

        #src_anchors = self.anchors[rois[:, 4].type(torch.cuda.LongTensor), :]
        src_anchors = self.anchors[rois[:, 4].long(), :]
        #src_anchors = torch.tensor(src_anchors, requires_grad=False, device=self.device)
        if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights
        bbox_x3d_dn = bbox_x3d_dn * widths.unsqueeze(0) + ctr_x.unsqueeze(0)
        bbox_y3d_dn = bbox_y3d_dn * heights.unsqueeze(0) + ctr_y.unsqueeze(0)
        bbox_z3d_dn = bbox_z3d_dn + src_anchors[:, 4].unsqueeze(0) 
        bbox_w3d_dn = torch.exp(bbox_w3d_dn) * src_anchors[:, 5].unsqueeze(0)
        bbox_h3d_dn = torch.exp(bbox_h3d_dn) * src_anchors[:, 6].unsqueeze(0)
        bbox_l3d_dn = torch.exp(bbox_l3d_dn) * src_anchors[:, 7].unsqueeze(0)
        bbox_ry3d_dn = src_anchors[:, 8].unsqueeze(0) + bbox_ry3d_dn

        bbox_z3d_dn_tar = bbox_z3d_tar * self.bbox_stds[0, 6] + self.bbox_means[0, 6]
        bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

        for bind in range(0, batch_size):

            #if not ((rmvs == False) & (igns == False)).any():
            #    continue
            if not any_val[bind]:
                continue

            #fg_inds = np.flatnonzero(labels_fg[bind])
            #bg_inds = np.flatnonzero(labels_bg[bind])
            #ign_inds = np.flatnonzero(labels_ign[bind])
            fg_inds  = torch.nonzero(labels_fg[bind].view(-1))
            bg_inds  = torch.nonzero(labels_bg[bind].view(-1))
            ign_inds = torch.nonzero(labels_ign[bind].view(-1))

            #if self.decouple:
            #    bbox_z3d_dn[bind] = bbox_z3d_dn[bind] * (p2[bind][1][1] / 1000.0)
            #    bbox_z3d_dn_tar[bind] = bbox_z3d_dn_tar[bind] * (p2[bind][1][1] / 1000.0)

            if fg_inds.shape[0] > 0 or ign_inds.shape[0] > 0:

                #logging.info('batch[{}]: {}fg_inds, {}bg_inds, {}ign_inds'.format(bind, fg_inds.shape, bg_inds.shape,ign_inds.shape))
                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if math.isinf(self.box_samples):
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois.shape[0]*self.box_samples * self.fg_fraction), fg_inds.shape[0])
                    bg_num = min(round(rois.shape[0]*self.box_samples - fg_num), bg_inds.shape[0])

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds]].view(-1)
                        #fg_score_ascend = (scores).argsort()
                        _, fg_score_ascend = torch.sort(scores, dim=0)
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds]].view(-1)
                        #bg_score_ascend = (scores).argsort()
                        _, bg_score_ascend = torch.sort(scores, dim=0)
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        #fg_inds = np.random.choice(fg_inds, fg_num, replace=False)
                        inds_tmp = torch.randperm(len(fg_inds))[:fg_num]
                        fg_inds = fg_inds[inds_tmp]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        #bg_inds = np.random.choice(bg_inds, bg_num, replace=False)
                        inds_tmp = torch.randperm(len(bg_inds))[:bg_num]
                        bg_inds = bg_inds[inds_tmp]

                labels_weight[bind, bg_inds] = BG_ENC
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1
                batch_fg[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                
                # if fg_num > 0:

                #     # compile deltas pred
                #     deltas_2d = bbox_2d[bind]
                #     #deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis ], bbox_y[bind, :, np.newaxis],
                #     #                       bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

                #     # compile deltas targets
                #     #deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                #     #                                bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                #     #                               axis=1)

                #     # move to gpu
                #     #deltas_2d_tar = torch.tensor(bbox_2d_tar, requires_grad=False, dtype=torch.float, device=self.device)
                #     deltas_2d_tar = bbox_2d_tar[bind]

                #     means = self.bbox_means[0, :]
                #     stds = self.bbox_stds[0, :]

                #     rois = rois.to(self.device)

                #     coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)
                #     coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)

                #     #logging.info('deltas_2d_tar[fg_inds]:\n{}'.format(deltas_2d_tar[fg_inds]))
                #     #logging.info('rois[fg_inds]:\n{}'.format(rois[fg_inds]))
                #     #logging.info('means:\n{}'.format(means))
                #     #logging.info('stds:\n{}'.format(stds))

                #     ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')
                #     #logging.info('ious_2d[bind, fg_inds]:\n{}'.format(ious_2d[bind, fg_inds]))
                #     #logging.info('coords_2d[bind, fg_inds]:\n{}'.format(coords_2d[fg_inds]))
                #     #logging.info('coords_2d_tar[bind, fg_inds]:\n{}'.format(coords_2d_tar[fg_inds]))

                #     bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                #     bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]

                #     #src_anchors = self.anchors[rois[fg_inds, 4].type(torch.cuda.LongTensor), :]
                #     src_anchors = self.anchors[rois[fg_inds, 4].cpu().int(), :]
                #     src_anchors = torch.tensor(src_anchors, requires_grad=False).type(torch.cuda.FloatTensor)
                #     if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

                #     #bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                #     #bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
                #     bbox_z3d_dn_fg = bbox_z3d_dn[bind, fg_inds]
                #     #bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
                #     #bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
                #     #bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
                #     bbox_ry3d_dn_fg = bbox_ry3d_dn[bind, fg_inds]

                #     # re-scale all 2D back to original
                #     #bbox_x3d_dn_fg /= imobj['scale_factor']
                #     #bbox_y3d_dn_fg /= imobj['scale_factor']

                #     #coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_y3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_z3d_dn_fg[np.newaxis,:]), dim=0)
                #     #coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)

                #     #coords_3d = torch.mm(p2_inv, coords_2d)
                #     #print('coords_3d:', coords_3d)

                #     #bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :]
                #     #bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :]
                #     #bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :]

                #     # absolute targets
                #     bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                #     bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                #     bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                #     bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]
                #     bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False).type(torch.cuda.FloatTensor)
                #     bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                #     coords_abs_z[bind, fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
                #     #logging.info('coords_abs_z[bind, fg_inds]:{}'.format(coords_abs_z[bind, fg_inds] ))
                #     #logging.info('bbox_z3d_dn_tar:{}'.format(bbox_z3d_dn_tar))
                #     #logging.info('bbox_z3d_dn_fg:{}'.format(bbox_z3d_dn_fg))
                #     coords_abs_ry[bind, fg_inds] = torch.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg)

            else:

                #bg_inds = np.arange(0, rois.shape[0])
                bg_inds = torch.arange(0, rois.shape[0])

                #if self.box_samples == np.inf: bg_num = len(bg_inds)
                if math.isinf(self.box_samples): bg_num = bg_inds.shape[0]
                else: bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), bg_inds.shape[0])

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds]].view(-1)
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        #bg_inds = np.random.choice(bg_inds, bg_num, replace=False)
                        inds_tmp = torch.randperm(len(bg_inds))[:bg_num]
                        bg_inds = bg_inds[inds_tmp]


                #labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC


            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active]]
            #logging.info('active:{}'.format(np.flatnonzero(active).shape))

        # end of for statement


        # ----------------------------------------
        # useful statistics
        #batch_fg_inds = np.nonzero(batch_fg)
        #batch_fg_inds = torch.nonzero(batch_fg, as_tuple=True)
        batch_fg_inds = torch.nonzero(batch_fg)
        batch_fg_inds = tuple((batch_fg_inds[:,0], batch_fg_inds[:, 1]))
        if batch_fg_inds[0].shape[0] > 0:

            # compile deltas pred
            #deltas_2d = bbox_2d
            #deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis ], bbox_y[bind, :, np.newaxis],
            #                       bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

            # compile deltas targets
            #deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
            #                                bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
            #                               axis=1)

            # move to gpu
            #deltas_2d_tar = torch.tensor(bbox_2d_tar, requires_grad=False, device=self.device)

            means = self.bbox_means[0, :]
            stds = self.bbox_stds[0, :]


            #print('bbox_2d.device:', bbox_2d.device) 
            #print('bbox_2d_tar.device:', bbox_2d_tar.device) 
            coords_2d = bbox_transform_inv_new(batch_rois, bbox_2d, means=means, stds=stds)
            coords_2d_tar = bbox_transform_inv_new(batch_rois, bbox_2d_tar, means=means, stds=stds)

            ious_2d[batch_fg_inds] = iou(coords_2d[batch_fg_inds], coords_2d_tar[batch_fg_inds], mode='list')

            #bbox_x3d_dn_fg = bbox_x3d_dn[batch_fg_inds]
            #bbox_y3d_dn_fg = bbox_y3d_dn[batch_fg_inds]

            #src_anchors = self.anchors[rois[fg_inds, 4].type(torch.cuda.LongTensor), :]
            src_anchors_inds = batch_rois[batch_fg_inds][:, 4].long()
            src_anchors = self.anchors[src_anchors_inds]
            #src_anchors = torch.tensor(src_anchors, requires_grad=False, device=self.device)
            if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

            #bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
            #bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
            bbox_z3d_dn_fg = bbox_z3d_dn[batch_fg_inds]
            #bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
            #bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
            #bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
            bbox_ry3d_dn_fg = bbox_ry3d_dn[batch_fg_inds]

            # re-scale all 2D back to original
            #bbox_x3d_dn_fg /= imobj['scale_factor']
            #bbox_y3d_dn_fg /= imobj['scale_factor']

            #coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_y3d_dn_fg[np.newaxis,:] * bbox_z3d_dn_fg[np.newaxis,:], bbox_z3d_dn_fg[np.newaxis,:]), dim=0)
            #coords_2d = torch.cat((coords_2d, torch.ones([1, coords_2d.shape[1]])), dim=0)

            #coords_3d = torch.mm(p2_inv, coords_2d)
            #print('coords_3d:', coords_3d)

            #bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :]
            #bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :]
            #bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :]

            # absolute targets
            # bbox_z3d_dn_tar = bbox_z3d_tar[batch_fg_inds] * self.bbox_stds[0, 6] + self.bbox_means[0, 6]
            bbox_z3d_dn_tar = bbox_z3d_dn_tar[batch_fg_inds]
            #bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, requires_grad=False, device=self.device)
            # bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

            bbox_ry3d_dn_tar = bbox_ry3d_tar[batch_fg_inds] * self.bbox_stds[0, 10] + self.bbox_means[0, 10]
            #bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False, device=self.device)
            bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

            coords_abs_z[batch_fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
            coords_abs_ry[batch_fg_inds] = torch.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg)
        # ----------------------------------------

        #fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        #bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        #fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        #bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        #fg_inds_unravel = torch.nonzero((labels > 0) & (labels != IGN_FLAG), as_tuple=True)
        #bg_inds_unravel = torch.nonzero((labels == 0) & (labels != IGN_FLAG), as_tuple=True)
        fg_inds_unravel = torch.nonzero((labels > 0) & (labels != IGN_FLAG))
        fg_inds_unravel = tuple((fg_inds_unravel[:, 0], fg_inds_unravel[:, 1]))
        bg_inds_unravel = torch.nonzero((labels == 0) & (labels != IGN_FLAG))
        bg_inds_unravel = tuple((bg_inds_unravel[:, 0], bg_inds_unravel[:, 1]))

        cls_pred = cls.argmax(dim=2)
        #logging.info('cls_pred:')
        #logging.info(cls_pred[fg_inds_unravel])
        #logging.info('labels:')
        #logging.info(labels[fg_inds_unravel])

        #logging.info('bg_pred:')
        #logging.info(cls_pred[bg_inds_unravel])
        #logging.info('bg_labels:')
        #logging.info(labels[bg_inds_unravel])

        if self.cls_2d_lambda and len(fg_inds_unravel[0]) > 0:
            acc_fg = torch.mean((cls_pred[fg_inds_unravel] == labels[fg_inds_unravel]).float())
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})

        if self.cls_2d_lambda and len(bg_inds_unravel) > 0:
            acc_bg = torch.mean((cls_pred[bg_inds_unravel] == labels[bg_inds_unravel]).float())
            stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        #fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        #bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        #active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        #fg_inds_unravel = torch.nonzero(labels_weight == FG_ENC, as_tuple=True)
        #bg_inds_unravel = torch.nonzero(labels_weight == BG_ENC, as_tuple=True)
        fg_inds_unravel = torch.nonzero(labels_weight == FG_ENC)
        fg_inds_unravel = tuple((fg_inds_unravel[:, 0], fg_inds_unravel[:, 1]))
        bg_inds_unravel = torch.nonzero(labels_weight == BG_ENC)
        bg_inds_unravel = tuple((bg_inds_unravel[:, 0], bg_inds_unravel[:, 1]))
        active_inds_unravel = tuple(torch.cat((fg_inds_unravel[i], bg_inds_unravel[i]), dim=0) 
                                    for i in range(len(bg_inds_unravel)))

        fg_num = len(fg_inds_unravel[0])
        bg_num = len(bg_inds_unravel[0])
        #logging.info('box weighting fg_num:{}, bg_num{}'.format(fg_num, bg_num))

        #logging.info('box weighting fg_num:{}, bg_num{}'.format(fg_num, bg_num))
        #logging.info('fg_inds:')
        #logging.info(str(list(fg_inds)))

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        #fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        #bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        #active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction /(1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights


        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        #labels = torch.tensor(labels, requires_grad=False, dtype=torch.long, device=self.device)
        labels = labels.view(-1)

        #labels_weight = torch.tensor(labels_weight, requires_grad=False, device=self.device, dtype=torch.float)
        labels_weight = labels_weight.view(-1)

        cls = cls.view(-1, cls.shape[2])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0
            #logging.info('cls active:{}'.format(np.flatnonzero(active).shape))
            #logging.info('cls active:{}'.format(np.flatnonzero(active)))

            if torch.any(active):

                loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                #logging.info('cls_loss:{}'.format(loss_cls))
                stats.append({'name': 'cls', 'val': loss_cls, 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if torch.sum(bbox_weights) > 0:

            #bbox_weights = torch.tensor(bbox_weights, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
            bbox_weights = bbox_weights.view(-1)

            active = bbox_weights > 0
            #logging.info('bbox active:{}'.format(np.flatnonzero(active).shape))
            #logging.info('bbox active:{}'.format(np.flatnonzero(active)))
            #logging.info('bbox_weights:{}'.format(bbox_weights[active]))

            if self.bbox_2d_lambda:

                # bbox loss 2d
                #bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                #bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                #bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)
                #bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False, device=self.device, dtype=torch.float).view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                #logging.info('bbox_2d_loss:{}'.format(bbox_2d_loss))
                #stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})


            if self.bbox_3d_lambda:

                # bbox loss 3d
                #bbox_x3d_tar  = torch.tensor(bbox_x3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_y3d_tar  = torch.tensor(bbox_y3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_z3d_tar  = torch.tensor(bbox_z3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_w3d_tar  = torch.tensor(bbox_w3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_h3d_tar  = torch.tensor(bbox_h3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_l3d_tar  = torch.tensor(bbox_l3d_tar, requires_grad=False, device=self.device).view(-1)
                #bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False, device=self.device).view(-1)

                bbox_x3d_tar  = bbox_x3d_tar.view(-1)
                bbox_y3d_tar  = bbox_y3d_tar.view(-1)
                bbox_z3d_tar  = bbox_z3d_tar.view(-1)
                bbox_w3d_tar  = bbox_w3d_tar.view(-1)
                bbox_h3d_tar  = bbox_h3d_tar.view(-1)
                bbox_l3d_tar  = bbox_l3d_tar.view(-1)
                bbox_ry3d_tar = bbox_ry3d_tar.view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_ry3d = bbox_ry3d[:, :].view(-1)

                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none')

                loss_bbox_x3d = (loss_bbox_x3d * bbox_weights[active]).mean()
                loss_bbox_y3d = (loss_bbox_y3d * bbox_weights[active]).mean()
                loss_bbox_z3d = (loss_bbox_z3d * bbox_weights[active]).mean()
                loss_bbox_w3d = (loss_bbox_w3d * bbox_weights[active]).mean()
                loss_bbox_h3d = (loss_bbox_h3d * bbox_weights[active]).mean()
                loss_bbox_l3d = (loss_bbox_l3d * bbox_weights[active]).mean()
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active]).mean()

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d + loss_bbox_ry3d)

                bbox_3d_loss *= self.bbox_3d_lambda

                loss += bbox_3d_loss
                #logging.info('bbox_3d_loss:{}'.format(bbox_3d_loss))
                stats.append({'name': 'bbox3d', 'val': bbox_3d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            #if self.bbox_3d_proj_lambda:

            #    # bbox loss 3d
            #    bbox_x3d_proj_tar = torch.tensor(bbox_x3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
            #    bbox_y3d_proj_tar = torch.tensor(bbox_y3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)
            #    bbox_z3d_proj_tar = torch.tensor(bbox_z3d_proj_tar, requires_grad=False).type(torch.FloatTensor).cuda().view(-1)

            #    bbox_x3d_proj = bbox_x3d_proj[:, :].view(-1)
            #    bbox_y3d_proj = bbox_y3d_proj[:, :].view(-1)
            #    bbox_z3d_proj = bbox_z3d_proj[:, :].view(-1)

            #    loss_bbox_x3d_proj = F.smooth_l1_loss(bbox_x3d_proj[active], bbox_x3d_proj_tar[active], reduction='none')
            #    loss_bbox_y3d_proj = F.smooth_l1_loss(bbox_y3d_proj[active], bbox_y3d_proj_tar[active], reduction='none')
            #    loss_bbox_z3d_proj = F.smooth_l1_loss(bbox_z3d_proj[active], bbox_z3d_proj_tar[active], reduction='none')

            #    loss_bbox_x3d_proj = (loss_bbox_x3d_proj * bbox_weights[active]).mean()
            #    loss_bbox_y3d_proj = (loss_bbox_y3d_proj * bbox_weights[active]).mean()
            #    loss_bbox_z3d_proj = (loss_bbox_z3d_proj * bbox_weights[active]).mean()

            #    bbox_3d_proj_loss = (loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj)

            #    bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

            #    loss += bbox_3d_proj_loss
            #    stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss, 'format': '{:0.4f}', 'group': 'loss'})

            coords_abs_z = coords_abs_z.view(-1)
            #logging.info('misc_z:{}'.format(coords_abs_z[active].mean()))
            stats.append({'name': 'z', 'val': coords_abs_z[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            coords_abs_ry = coords_abs_ry.view(-1)
            #logging.info('misc_ry:{}'.format(coords_abs_ry[active].mean()))
            stats.append({'name': 'ry', 'val': coords_abs_ry[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            ious_2d = ious_2d.view(-1)
            #logging.info('ious_2d:{}'.format(ious_2d[active].mean()))
            stats.append({'name': 'iou', 'val': ious_2d[active].mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                iou_2d_loss = iou_2d_loss.mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                #logging.info('iou_2d_loss:{}'.format(iou_2d_loss))
                stats.append({'name': 'iou', 'val': iou_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})
            stats.append({'name': 'ttloss', 'val': loss, 'format': '{:0.4f}', 'group': 'loss'})


        return loss, stats
