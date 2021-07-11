# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import math

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *
from lib.dataloader import Kitti_Dataset_torch  #, Kitti_Dataset_dali


def print_grad(module, grad_input, grad_output):
    print(module)
    for grad in grad_output:
        print('output grad_mean:', torch.mean(grad))
        print('output grad_max:', torch.max(grad))
        print('output grad_min:', torch.min(grad))
        where = torch.where(grad != 0)
        print('output grad', grad[where])

    for grad in grad_input:
        print('input grad_mean:', torch.mean(grad))
        print('input grad_max:', torch.max(grad))
        print('input grad_min:', torch.min(grad))
        where = torch.where(grad != 0)
        print('input grad', grad[where])
    #print('grad_output:', grad_output)
    #print('grad_input:', grad_input)

feats_in = []
feats_out = []
def record_feats(module, in_feat, out_feat):

    if len(feats_out) == 0:
        feats_out.append(out_feat[0, 0:1, :, :])

def gen_eval(eval_epoch, max_epoch):
    epochs = []
    epoch = int(eval_epoch)
    epochs.append(epoch)

    while(epoch < max_epoch):
       eval_epoch = int(max(eval_epoch*0.5, 1))
       epoch += eval_epoch
       epochs.append(epoch)
    
    return epochs

        

def main(args):

    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    #opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = args.config
    restore = args.restore

    # read opts
    #for opt, arg in opts:

    #    if opt in ('--config'): conf_name = arg
    #    if opt in ('--restore'): restore = int(arg)

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------

    conf = init_config(conf_name)
    if restore:
        timestamp = args.restore_time
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = init_training_paths(args.exp_name, timestamp, conf)

    writer = SummaryWriter(log_dir= paths.logs)

    init_torch(conf)
    init_log_file(paths.logs)


    # defaults
    start_epoch = 0
    disp_tracker = edict()
    epoch_tracker = edict()
    iterator = None
    has_visdom = None

    if conf.pre_compute_target:
        dataset = Kitti_Dataset_torch(conf, paths)
        dataset_val = Kitti_Dataset_torch(conf, paths, phase='validation')
        iter_per_epoch = math.ceil(len(dataset) / conf.batch_size)
    else:
        dataset = Dataset(conf, paths.data, paths.data_cache)
        dataset_val = Dataset(conf, paths.data, paths.data_cache, phase='validation')
        iter_per_epoch = math.ceil(len(dataset) / conf.batch_size)


    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration
    pickle_write(os.path.join(paths.output, '{}.conf'.format(conf_name)), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    rpn_net, optimizer = init_training_model(conf, paths.output)

    # setup loss
    if conf.pre_compute_target:
        criterion_det = RPN_3D_loss_smp(conf)
    else:
        criterion_det = RPN_3D_loss(conf)

    # resume training
    if restore:
        print('restore from iter {}'.format(restore))
        start_epoch = restore
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore, conf.back_bone)
    # custom pretrained network
    elif 'pretrained' in conf:
        load_weights(rpn_net, conf.pretrained)


    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    optimizer.zero_grad()

    start_time = time()

    # -----------------------------------------
    # train
    # -----------------------------------------
    conf.max_iter = (conf.max_epoch) * iter_per_epoch
    epoch = 0
    best_iou = 0
    iteration = start_epoch * iter_per_epoch
    conf.eval_epochs = gen_eval(conf.eval_epoch, conf.max_epoch)
    for epoch in range(start_epoch, conf.max_epoch):
    #with trange(start_epoch, conf.max_epoch) as t_epoch:
    #    for epoch in t_epoch:

        #time_1 = time()
        #iterator, images, imobjs, next_epoch = next_iteration(dataset.loader, iterator, iteration)
        #for images, imobjs in tqdm(dataset.loader):
        #with tqdm(dataset.loader) as t_batch:
        #for images, imobjs in t_batch:
        for batch in dataset.loader:

            #images = images.cuda()
            if conf.pre_compute_target:
                images = batch['input']
                imobjs = batch['target']
                #logging.info('{}'.format(imobjs['meta']['id']))
            else:
                images, imobjs = batch
                #logging.info('{}'.format([imobj['id'] for imobj in imobjs]))

            #  learning rate
            iteration += 1
            lr = adjust_lr(conf, optimizer, iteration)
            #lr = adjust_lr_per_epoch(conf, optimizer, epoch+1)

            # train mode
            rpn_net.train()

            # forward
            #logging.info('images max:{}, min{}'.format(torch.max(images), torch.min(images)))
            cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images)

            # loss
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)
            total_loss = det_loss.cpu().detach().numpy()
            stats = det_stats

            # backprop
            if total_loss > 0:

                optimizer.zero_grad()
                det_loss.backward()
                # batch skip, simulates larger batches by skipping gradient step
                if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                    optimizer.step()

            # keep track of stats
            compute_stats(disp_tracker, stats)
            compute_stats(epoch_tracker, stats)
            # -----------------------------------------
            # display
            # -----------------------------------------
            post = dict()
            post['lr'] = '{:.2e}'.format(lr)
            post['loss'] = '{:.3f}'.format(total_loss)
            # for stat in stats:
            #     if stat['group'] == 'acc' or stat['group'] == 'misc':
            #         name = stat['name']
            #         val = stat['val']
            #         post[name] = '{:.3f}'.format(val)
            for key, val in epoch_tracker.items():
                group, name = key.split('_')
                if group == 'acc' or group == 'misc' or name =='ttloss':
                    post[name] = '{:.3f}'.format(np.mean(val))


            if (iteration + 1) % conf.display_iter == 0 :

                writer.add_scalar("Train/lr", lr, iteration+1)
                post = "[Train] {}: lr: {:.3e}".format(iteration+1, lr)
                for key, val in disp_tracker.items():
                    group, name = key.split('_')
                    writer.add_scalar( "Train/"+key, np.mean(val), iteration+1)
                    post = post + "  {}_{}: {:.3f}".format(group, name, np.mean(val))
                    
                logging.info(post)

                disp_tracker = edict()

        logging.info('Epoch:{}'.format(epoch))
        epoch_post = dict()
        for key, val in epoch_tracker.items():
            group, name = key.split('_')
            if group == 'acc' or group == 'misc' or name =='ttloss':
                epoch_post[name] = '{:.3f}'.format(np.mean(val))
                logging.info("{}/{}: {:.3f}".format(group, name, np.mean(val)))
        
        epoch_tracker = edict()
        # -----------------------------------------
        # test network
        # -----------------------------------------
        
        if int(epoch+1) in conf.eval_epochs:
            # store checkpoint
            save_checkpoint(optimizer, rpn_net, paths.weights, epoch+1, conf.back_bone)

            if conf.do_test:

                with torch.no_grad():
                    # eval mode
                    rpn_net.eval()

                    # necessary paths
                    results_path = os.path.join(paths.results, 'results_{}'.format(epoch+1))

                    # -----------------------------------------
                    # test kitti
                    # -----------------------------------------
                    if conf.test_protocol.lower() == 'kitti':
                        print('testing')
                        # delete and re-make
                        results_path = os.path.join(results_path, 'data')
                        mkdir_if_missing(results_path, delete_if_exist=True)

                        iou_3d = test_kitti_3d(dataset_val, rpn_net, conf, results_path, paths.data, writer=writer)
                        #prob, bbox_2d, bbox_3d, feat_size = outputs
                        #val_loss, val_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

                        # save the best model
                        iou_3d = np.array(iou_3d)
                        mean_iou = np.mean(iou_3d)
                        if mean_iou > best_iou:
                            save_checkpoint(optimizer, rpn_net, paths.weights, 'best', conf.back_bone)
                            best_iou = mean_iou

                    else:
                        logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))


                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)
    writer.close()
    if best_iou > 1:
        output_dir = paths.output + '_{}'.format(best_iou)
        cmd = 'mv {} {}'.format(paths.output, output_dir)
        os.system(cmd)


# run from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="kitti_3d_multi_main", 
                        help="which config file to use, kitti_3d_multi_main | kitti_3d_multi_warmup")
    parser.add_argument("--restore", type=int, default=None,
                        help="resuming training at which epoch")
    parser.add_argument("--restore_time", type=str, default=None,
                        help="resuming training at which epoch")
    parser.add_argument("--exp_name", type=str, default='3dbox',
                        help="give a name for the experiment")
    args = parser.parse_args()
    main(args)
