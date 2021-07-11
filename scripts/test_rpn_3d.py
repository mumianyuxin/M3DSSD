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

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from lib.dataloader import Kitti_Dataset_torch, Kitti_Dataset_dali
# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path    = 'path to config file used when training'
weights_path = 'path to weight of the model after training'

# load config
conf = edict(pickle_read(conf_path))


val_train = False
paths = init_training_paths('base', '0000' ,conf)
if val_train:
    results_path = os.path.join('output', 'tmp_results_train', 'data')
    dataset_test = Kitti_Dataset_torch(conf, paths, phase='val_train')
else:
    results_path = os.path.join('output', 'tmp_results', 'data')
    dataset_test = Kitti_Dataset_torch(conf, paths, phase='test')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

print(pretty_print('conf', conf))

# defaults
init_torch(conf)

# setup network
net = import_module('model.' + conf.model).build(conf, 'test')
net.eval()
if conf.use_cuda:
    net = torch.nn.DataParallel(net)
# load weights
weights = torch.load(weights_path)
net.load_state_dict(weights)

# switch modes for evaluation
net.eval()

test_kitti_3d(dataset_test, net, conf, results_path, paths.data, use_log=False, val_train=val_train)