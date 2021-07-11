"""
This file is meant to contain all functions of the detective framework
which are "specific" to the framework but generic among experiments.

For example, all the experiments need to initialize configs, training models,
log stats, display stats, and etc. However, these functions are generally fixed
to this framework and cannot be easily transferred in other projects.
"""

# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from copy import copy
import importlib
import random
#import visdom
import torch
import shutil
import sys
import os
import cv2
import math

# stop python from writing so much bytecode
sys.dont_write_bytecode = True

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.util import *


def init_config(conf_name):
    """
    Loads configuration file, by checking for the conf_name.py configuration file as
    ./config/<conf_name>.py which must have function "Config".

    This function must return a configuration dictionary with any necessary variables for the experiment.
    """

    conf = importlib.import_module('config.' + conf_name).Config()

    return conf


def init_training_model(conf, cache_folder):
    """
    This function is meant to load the training model and optimizer, which expects
    ./model/<conf.model>.py to be the pytorch model file.

    The function copies the model file into the cache BEFORE loading, for easy reproducibility.
    """

    src_path = os.path.join('.', 'model', conf.model + '.py')
    #src_path_1 = os.path.join('.', 'models', 'pose_dla_dcn.py')
    dst_path = os.path.join(cache_folder, conf.model + '.py')
    #dst_path_1 = os.path.join(cache_folder, 'pose_dla_dcn.py')

    # (re-) copy the pytorch model file
    if os.path.exists(dst_path): os.remove(dst_path)
    #if os.path.exists(dst_path_1): os.remove(dst_path_1)
    shutil.copyfile(src_path, dst_path)
    #shutil.copyfile(src_path_1, dst_path_1)

    # load and build
    network = absolute_import(dst_path)
    network = network.build(conf, 'train')

    # multi-gpu
    if conf.use_cuda:
        network = torch.nn.DataParallel(network)

    # load SGD
    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        mo = conf.momentum
        wd = conf.weight_decay

        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=mo, weight_decay=wd)

    # load adam
    elif conf.solver_type.lower() == 'adam':

        lr = conf.lr
        wd = conf.weight_decay

        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)

    # load adamax
    elif conf.solver_type.lower() == 'adamax':

        lr = conf.lr
        wd = conf.weight_decay

        optimizer = torch.optim.Adamax(network.parameters(), lr=lr, weight_decay=wd)


    return network, optimizer


def adjust_lr(conf, optimizer, iter):
    """
    Adjusts the learning rate of an optimizer according to iteration and configuration,
    primarily regarding regular SGD learning rate policies.

    Args:
        conf (dict): configuration dictionary
        optimizer (object): pytorch optim object
        iter (int): current iteration
    """

    if 'batch_skip' in conf and ((iter + 1) % conf.batch_skip) > 0: return

    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        lr_steps = conf.lr_steps
        max_iter = conf.max_iter
        lr_policy = conf.lr_policy
        lr_target = conf.lr_target

        if lr_steps:
            steps = np.array(lr_steps) * max_iter
            total_steps = steps.shape[0]
            step_count = np.sum((steps - iter) <= 0)

        else:
            total_steps = max_iter
            step_count = iter

        # perform the exact number of steps needed to get to lr_target
        if lr_policy.lower() == 'step':
            scale = (lr_target / lr) ** (1 / total_steps)
            lr *= scale ** step_count

        # compute the scale needed to go from lr --> lr_target
        # using a polynomial function instead.
        elif lr_policy.lower() == 'poly':

            if step_count < int(total_steps*conf.warmup):
                scale = step_count/(total_steps*conf.warmup)
                lr = scale * conf.lr
            else:
                power = 0.9
                scale = total_steps / (1 - (lr_target / lr) ** (1 / power))
                lr *= (1 - step_count / scale) ** power

        elif lr_policy.lower() == 'cos':

            if step_count < int(max_iter*conf.warmup):
                scale = step_count/(max_iter*conf.warmup)
                lr = scale * conf.lr
            else:
                step_count -= int(max_iter*conf.warmup)
                max_iter -= int(max_iter*conf.warmup)
                scale = step_count/max_iter
                lr = conf.lr_target + 0.5 * (conf.lr-conf.lr_target) * (1 + math.cos(scale*math.pi))

        else:
            raise ValueError('{} lr_policy not understood'.format(lr_policy))

        # update the actual learning rate
        for gind, g in enumerate(optimizer.param_groups):
            g['lr'] = lr
    else:
        lrs = []
        for gind, g in enumerate(optimizer.param_groups):
            lrs.append(g['lr'])
        lrs = np.array(lrs)
        lr = np.mean(lrs)
        #print(lr)
    return  lr

def adjust_lr_per_epoch(conf, optimizer, epoch):
    """
    Adjusts the learning rate of an optimizer according to iteration and configuration,
    primarily regarding regular SGD learning rate policies.

    Args:
        conf (dict): configuration dictionary
        optimizer (object): pytorch optim object
        iter (int): current iteration
    """

    if 'batch_skip' in conf and ((iter + 1) % conf.batch_skip) > 0: return

    if conf.solver_type.lower() == 'sgd':

        lr = conf.lr
        lr_steps = conf.lr_steps
        max_epoch = conf.max_epoch
        lr_policy = conf.lr_policy
        lr_target = conf.lr_target

        if lr_steps:
            steps = np.array(lr_steps) * max_epoch
            total_steps = steps.shape[0]
            step_count = np.sum((steps - iter) <= 0)

        else:
            total_steps = max_epoch
            step_count = epoch

        # perform the exact number of steps needed to get to lr_target
        if lr_policy.lower() == 'step':
            scale = (lr_target / lr) ** (1 / total_steps)
            lr *= scale ** step_count

        # compute the scale needed to go from lr --> lr_target
        # using a polynomial function instead.
        elif lr_policy.lower() == 'poly':

            power = 0.9
            scale = total_steps / (1 - (lr_target / lr) ** (1 / power))
            lr *= (1 - step_count / scale) ** power

        elif lr_policy.lower() == 'cos':

            if step_count < int(total_steps *0.05):
                scale = step_count/(total_steps *0.05)
                lr = scale * conf.lr
            else:
                step_count -= int(total_steps *0.05)
                total_steps  -= int(total_steps *0.05)
                scale = step_count/total_steps 
                lr = conf.lr_target + 0.5 * (conf.lr-conf.lr_target) * (1 + math.cos(scale*math.pi))

        else:
            raise ValueError('{} lr_policy not understood'.format(lr_policy))

        # update the actual learning rate
        for gind, g in enumerate(optimizer.param_groups):
            g['lr'] = lr
    else:
        lrs = []
        for gind, g in enumerate(optimizer.param_groups):
            lrs.append(g['lr'])
        lrs = np.array(lrs)
        lr = np.mean(lrs)
        #print(lr)
    return  lr



def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        elif data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:4], torch.unsqueeze(box_b[:, 2:4], dim=1))
            min_xy = torch.max(box_a[:, 0:2], torch.unsqueeze(box_b[:, 0:2], dim=1))
            inter = torch.clamp((max_xy - min_xy), min=0, max=None)
        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou3d(corners_3d_b1, corners_3d_b2, vol):

    corners_3d_b1 = copy.copy(corners_3d_b1)
    corners_3d_b2 = copy.copy(corners_3d_b2)

    corners_3d_b1 = corners_3d_b1.T
    corners_3d_b2 = corners_3d_b2.T

    y_min_b1 = np.min(corners_3d_b1[:, 1])
    y_max_b1 = np.max(corners_3d_b1[:, 1])
    y_min_b2 = np.min(corners_3d_b2[:, 1])
    y_max_b2 = np.max(corners_3d_b2[:, 1])
    y_intersect = np.max([0, np.min([y_max_b1, y_max_b2]) - np.max([y_min_b1, y_min_b2])])

    # set Z as Y
    corners_3d_b1[:, 1] = corners_3d_b1[:, 2]
    corners_3d_b2[:, 1] = corners_3d_b2[:, 2]

    polygon_order = [7, 2, 3, 6, 7]
    box_b1_bev = Polygon([list(corners_3d_b1[i][0:2]) for i in polygon_order])
    box_b2_bev = Polygon([list(corners_3d_b2[i][0:2]) for i in polygon_order])

    intersect_bev = box_b2_bev.intersection(box_b1_bev).area
    intersect_3d = y_intersect * intersect_bev

    iou_bev = intersect_bev / (box_b2_bev.area + box_b1_bev.area - intersect_bev)
    iou_3d = intersect_3d / (vol - intersect_3d)

    return iou_bev, iou_3d


def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # torch.Tensor
        if data_type == torch.Tensor:
            inter = intersect(box_a, box_b, data_type=data_type)
            area_a = ((box_a[:, 2] - box_a[:, 0]) *
                      (box_a[:, 3] - box_a[:, 1]))
            area_b = ((box_b[:, 2] - box_b[:, 0]) *
                      (box_b[:, 3] - box_b[:, 1]))
            union = torch.unsqueeze(area_a, 0) + torch.unsqueeze(area_b, 1) - inter

            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            inter = intersect(box_a, box_b, data_type=data_type)
            area_a = ((box_a[:, 2] - box_a[:, 0]) *
                      (box_a[:, 3] - box_a[:, 1]))
            area_b = ((box_b[:, 2] - box_b[:, 0]) *
                      (box_b[:, 3] - box_b[:, 1]))
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter

            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / (union+ 1e-8)

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou_ign(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of overap of box_b has within box_a, which is handy for dealing with ignore regions.
    Hence, assume that box_b are ignore regions and box_a are anchor boxes, then we may want to know how
    much overlap the anchors have inside of the ignore regions (hence ignore area_b!)

    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))

        # torch and numpy have different calls for transpose
        if data_type == torch.Tensor:
            union = torch.unsqueeze(area_a, dim=0) + torch.unsqueeze(area_b, dim=1) * 0 - inter * 0
            return (inter / union).permute(1, 0)
        elif data_type == np.ndarray:
            union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) * 0 - inter * 0
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    else:
        raise ValueError('unknown mode {}'.format(mode))


def freeze_layers(network, blacklist=None, whitelist=None, verbose=False):

    if blacklist is not None:

        for name, param in network.named_parameters():

            if not any([allowed in name for allowed in blacklist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False

        for name, module in network.named_modules():
            if not any([allowed in name for allowed in blacklist]):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

    if whitelist is not None:

        for name, param in network.named_parameters():

            if any([banned in name for banned in whitelist]):
                if verbose:
                    logging.info('freezing {}'.format(name))
                param.requires_grad = False
            #else:
            #    logging.info('NOT freezing {}'.format(name))

        for name, module in network.named_modules():
            if any([banned in name for banned in whitelist]):
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()


def load_weights(model, path, src_weights=None, remove_module=False):
    """
    Simply loads a pytorch models weights from a given path.
    """
    print(' Loading weights of pretrained model:', path)
    dst_weights = model.state_dict()
    if not path is None:
        src_weights = torch.load(path)

    dst_keys = list(dst_weights.keys())
    src_keys = list(src_weights.keys())

    if remove_module:

        # copy keys without module
        for key in src_keys:
            src_weights[key.replace('module.', '')] = src_weights[key]
            del src_weights[key]
        src_keys = list(src_weights.keys())

        # remove keys not in dst
        for key in src_keys:
            if key not in dst_keys: del src_weights[key]

    else:

        # remove keys not in dst
        for key in src_keys:
            if key not in dst_keys: del src_weights[key]

        # add keys not in src
        for key in dst_keys:
            if key not in src_keys: src_weights[key] = dst_weights[key]

    model.load_state_dict(src_weights)


def log_stats(tracker, iteration, start_time, start_iter, max_iter, skip=1):
    """
    This function writes the given stats to the log / prints to the screen.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    display_str = 'iter: {}'.format((int((iteration + 1)/skip)))

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter, max_iter - start_iter)

    # cycle through all tracks
    last_group = ''
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:

            # compute mean
            meanval = np.mean(tracker[key])

            # get properties
            format = tracker[key + '_obj'].format
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # logic to have the string formatted nicely
            # basically roughly this format:
            #   iter: {}, group_1 (name: val, name: val), group_2 (name: val), dt: val, eta: val
            if last_group != group and last_group == '':
                display_str += (', {} ({}: ' + format).format(group, name, meanval)

            elif last_group != group:
                display_str += ('), {} ({}: ' + format).format(group, name, meanval)

            else:
                display_str += (', {}: ' + format).format(name, meanval)

            last_group = group

    # append dt and eta
    display_str += '), dt: {:0.2f}, eta: {}'.format(dt, time_str)

    # log
    logging.info(display_str)


def display_stats(vis, tracker, iteration, start_time, start_iter, max_iter, conf_name, conf_pretty, skip=1):
    """
    This function plots the statistics using visdom package, similar to the log_stats function.
    Also, computes the estimated time arrival (eta) for completion and (dt) delta time per iteration.

    Args:
        vis (visdom): the main visdom session object
        tracker (array): dictionary array tracker objects. See below.
        iteration (int): the current iteration
        start_time (float): starting time of whole experiment
        start_iter (int): starting iteration of whole experiment
        max_iter (int): maximum iteration to go to
        conf_name (str): experiment name used for visdom display
        conf_pretty (str): pretty string with ALL configuration params to display

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # compute eta
    time_str, dt = compute_eta(start_time, iteration - start_iter, max_iter - start_iter)

    # general info
    info = 'Experiment: <b>{}</b>, Eta: <b>{}</b>, Time/it: {:0.2f}s\n'.format(conf_name, time_str, dt)
    info += conf_pretty

    # replace all newlines and spaces with line break <br> and non-breaking spaces &nbsp
    info = info.replace('\n', '<br>')
    info = info.replace(' ', '&nbsp')

    # pre-formatted html tag
    info = '<pre>' + info + '</pre'

    # update the info window
    vis.text(info, win='info', opts={'title': 'info', 'width': 500, 'height': 350})

    # draw graphs for each track
    for key in sorted(tracker.keys()):

        if type(tracker[key]) == list:
            meanval = np.mean(tracker[key])
            group = tracker[key + '_obj'].group
            name = tracker[key + '_obj'].name

            # new data point
            vis.line(X=np.array([(iteration + 1)]), Y=np.array([meanval]), win=group, name=name, update='append',
                     opts={'showlegend': True, 'title': group, 'width': 500, 'height': 350,
                           'xlabel': 'iteration'})


def compute_stats(tracker, stats):
    """
    Copies any arbitary statistics which appear in 'stats' into 'tracker'.
    Also, for each new object to track we will secretly store the objects information
    into 'tracker' with the key as (group + name + '_obj'). This way we can retrieve these properties later.

    Args:
        tracker (array): dictionary array tracker objects. See below.
        stats (array): dictionary array tracker objects. See below.

    A tracker object is a dictionary with the following:
        "name": the name of the statistic being tracked, e.g., 'fg_acc', 'abs_z'
        "group": an arbitrary group key, e.g., 'loss', 'acc', 'misc'
        "format": the python string format to use (see official str format function in python), e.g., '{:.2f}' for
                  a float with 2 decimal places.
    """

    # through all stats
    for stat in stats:

        # get properties
        name = stat['name']
        group = stat['group']
        val = stat['val']

        # convention for identificaiton
        id = group +'_'+ name

        # init if not exist?
        if not (id in tracker): tracker[id] = []

        # convert tensor to numpy
        if type(val) == torch.Tensor:
            val = val.cpu().detach().numpy()

        # store
        tracker[id].append(val)

        # store object info
        #obj_id = id + '_obj'
        #if not (obj_id in tracker):
        #    stat.pop('val', None)
        #    tracker[id + '_obj'] = stat


def next_iteration(loader, iterator, iteration=0):
    """
    Loads the next iteration of 'iterator' OR makes a new epoch using 'loader'.

    Args:
        loader (object): PyTorch DataLoader object
        iterator (object): python in-built iter(loader) object
    """
    next_epoch = 0
    # create if none
    if iterator == None: iterator = iter(loader)

    # next batch
    try:
        #images, imobjs = next(iterator)
        batch = next(iterator)

    # new epoch / shuffle
    except StopIteration:
        next_epoch = 1
        iterator = iter(loader)
        #images, imobjs = next(iterator)
        batch = next(iterator)

    return iterator, batch, next_epoch


def init_training_paths(name, timestamp, conf, use_tmp_folder=None):
    """
    Simple function to store and create the relevant paths for the project,
    based on the base = current_working_dir (cwd). For this reason, we expect
    that the experiments are run from the root folder.

    data    =  ./data
    output  =  ./output/<name>/time/
    weights =  ./output/<name>/time/weights
    results =  ./output/<name>/time/results
    logs    =  ./output/<name>/time/log

    Args:
        conf_name (str): configuration experiment name (used for storage into ./output/<conf_name>)
    """

    # make paths
    paths = edict()
    paths.base = os.getcwd()
    paths.data = os.path.join(paths.base, 'data')
    if conf.mc:
        paths.data_cache = os.path.join(paths.base, 'data_cache', conf.datasets_train[0]['name']+'_mc')
    else:
        paths.data_cache = os.path.join(paths.base, 'data_cache', conf.datasets_train[0]['name'])
    
    if conf.anchor_ratios[2] == 2.0:
        paths.data_cache = paths.data_cache + '_2'
    paths.output = os.path.join(paths.base, 'output', name, timestamp)
    paths.weights = os.path.join(paths.output, 'weights')
    paths.logs = os.path.join(paths.output, 'log')

    if use_tmp_folder: paths.results = os.path.join(paths.base, '.tmp_results', name, 'results')
    else: paths.results = os.path.join(paths.output, 'results')

    # make directories
    mkdir_if_missing(paths.output)
    mkdir_if_missing(paths.data_cache)
    mkdir_if_missing(paths.logs)
    mkdir_if_missing(paths.weights)
    mkdir_if_missing(paths.results)

    return paths


def init_torch(conf):
    """
    Initializes the seeds for ALL potential randomness, including torch, numpy, and random packages.

    Args:
        rng_seed (int): the shared random seed to use for numpy and random
        cuda_seed (int): the random seed to use for pytorch's torch.cuda.manual_seed_all function
    """

    # default tensor
    conf.use_cuda = torch.cuda.is_available()
    if conf.use_cuda:
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        conf.device = torch.device('cuda')
    else:
        #torch.set_default_tensor_type('torch.FloatTensor')
        conf.device = torch.device('cpu')

    # seed everything
    torch.manual_seed(conf.rng_seed)
    np.random.seed(conf.rng_seed)
    random.seed(conf.rng_seed)
    torch.cuda.manual_seed_all(conf.cuda_seed)
    torch.cuda.manual_seed(conf.cuda_seed)

    # make the code deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_visdom(conf_name, visdom_port):
    """
    Simply initializes a visdom session (if possible) then closes all windows within it.
    If there is no visdom server running (externally), then function will return 'None'.
    """
    try:
        vis = visdom.Visdom(port=visdom_port, env=conf_name)
        vis.close(env=conf_name, win=None)

        if vis.socket_alive:
            return vis
        else:
            return None

    except:
        return None


def check_tensors():
    """
    Checks on tensors currently loaded within PyTorch
    for debugging purposes only (esp memory leaks).
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device, obj.shape)
        except:
            pass


def resume_checkpoint(optim, model, weights_dir, iteration, net_name):
    """
    Loads the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath = checkpoint_names(weights_dir, iteration, net_name)

    model.load_state_dict(torch.load(modelpath))
    optim.load_state_dict(torch.load(optimpath))


def save_checkpoint(optim, model, weights_dir, iteration, net_name):
    """
    Saves the optimizer and model pair given the current iteration
    and the weights storage directory.
    """

    optimpath, modelpath = checkpoint_names(weights_dir, iteration, net_name)

    torch.save(optim.state_dict(), optimpath)
    torch.save(model.state_dict(), modelpath)

    return modelpath, optimpath


def checkpoint_names(weights_dir, iteration, net_name):
    """
    Single function to determine the saving format for
    resuming and saving models/optim.
    """

    optimpath = os.path.join(weights_dir, 'optim_{}_{}_pkl'.format(net_name, iteration))
    modelpath = os.path.join(weights_dir, 'model_{}_{}_pkl'.format(net_name, iteration))

    return optimpath, modelpath


def print_weights(model):
    """
    Simply prints the weights for the model using the mean weight.
    This helps keep track of frozen weights, and to make sure
    they initialize as non-zero, although there are edge cases to
    be weary of.
    """

    # find max length
    max_len = 0
    for name, param in model.named_parameters():
        name = str(name).replace('module.', '')
        if (len(name) + 4) > max_len: max_len = (len(name) + 4)

    # print formatted mean weights
    for name, param in model.named_parameters():
        mdata = np.abs(torch.mean(param.data).item())
        name = str(name).replace('module.', '')

        logging.info(('{0:' + str(max_len) + '} {1:6} {2:6}')
                     .format(name, 'mean={:.4f}'.format(mdata), '    grad={}'.format(param.requires_grad)))

