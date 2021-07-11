# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
import argparse

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from lib.dataloader import Kitti_Dataset_torch
from lib.imdb_util import *
from scripts.train_rpn_3d import gen_eval

def eval_single(net, dataset, weights_path, conf, results_path, writer):
    
    # make directory
    mkdir_if_missing(results_path, delete_if_exist=True)

    # load weights
    load_weights(net, weights_path, remove_module=True)
    
    # switch modes for evaluation
    net.cuda()
    net.eval()
    
    test_kitti_3d(dataset, net, conf, results_path, paths.data, use_log=True, phase='validation', writer=writer)

def repeat_eval(paths, conf, exp_name):

    import time
    writer = SummaryWriter(log_dir= paths.logs)
    init_log_file(paths.logs)

    # training network
    init_torch(conf)
    dst_path = os.path.join(paths.output, conf.model + '.py')
    # load and build
    network = absolute_import(dst_path)
    network = network.build(conf, 'test')

    # dataset
    dataset_test = Kitti_Dataset_torch(conf, paths, phase='validation')

    for epoch in conf.eval_epochs:

        if conf.restore:
            if epoch != conf.res_epoch:
                continue
            else:
                conf.restore = False
        print('[{}]: waiting for epoch {} model weights...'.format(exp_name, epoch))
        optimpath, modelpath = checkpoint_names(paths.weights, epoch, conf.back_bone)
        while True:
            if os.path.exists(modelpath):
                print('testing of epoch {}:'.format(epoch))
                results_path = os.path.join(paths.results, 'results_{}'.format(epoch))
                eval_single(network, dataset_test, modelpath, conf, results_path, writer)
                break

            else:
                time.sleep(60)
    writer.close()

def newest_dir(root_dir):
    lists = os.listdir(root_dir)                                    # 列出目录的下所有文件和文件夹保存到lists
    lists.sort(key=lambda fn: os.path.getmtime(root_dir + "/" + fn))  # 按时间排序
    return lists[-1]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--exp_time', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--restore', type=bool, default=False)
    args = parser.parse_args()

    
    root_dir = os.path.join('./output', args.exp_name)
    if args.exp_time is None:
        exp_time = newest_dir(root_dir) 
    else:
        exp_time = args.exp_time
    root_dir = os.path.join(root_dir, exp_time)

    file_list = os.listdir(root_dir)
    for f in file_list:
        _, tail = os.path.splitext(f)
        if tail == '.conf':
            break

    print(f)
    conf_path = os.path.join(root_dir, f)
    conf = edict(pickle_read(conf_path))
    conf.restore = args.restore
    conf.res_epoch = args.epoch
    conf.hill_climbing=True
    if 'datasets_validation' not in conf:
        conf.datasets_validation = conf.datasets_test

    if args.epoch and not args.restore:
        conf.eval_epochs = [args.epoch]
    else:
        conf.eval_epochs = gen_eval(conf.eval_epoch, conf.max_epoch)

    paths = init_training_paths(args.exp_name, exp_time, conf)
    if args.tmp:
        paths.results = os.path.join('./output', 'tmp_results')

    print(pretty_print('conf', conf))

    print(pretty_print('paths', paths))

    repeat_eval(paths, conf, args.exp_name)
