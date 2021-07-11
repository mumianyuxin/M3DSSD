import torch, math
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)

import threading
from torch.multiprocessing import Event
#from torch._six import queue
import queue

#from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
#from nvidia.dali.pipeline import Pipeline
#import nvidia.dali.ops as ops
#import nvidia.dali.types as types
import cv2

from lib.util import *
from lib.rpn_util import *
from lib.imdb_util import *
from lib.augmentations import *
from lib.core import *

from copy import deepcopy

'''
class HybridTrainPipe(Pipeline):
    """
    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    shard_id (int, optional, default = 0) – Id of the part to read
    shard_num (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    """

    def __init__(self, batch_size, data_dir, file_list, num_threads, device_id, size_y, size_x,
                 mean, std, shard_id=0, shard_num=1, shuffle=True, seed=-1):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)

        # Enabling read_ahead slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, file_list=file_list, shard_id=shard_id, num_shards=shard_num,
                                    random_shuffle=shuffle, seed=seed)

        # Let user decide which pipeline works best with the chosen model
        self.decode_device = "mixed"
        self.dali_device = "gpu"

        self.cmn = ops.CropMirrorNormalize(device=self.dali_device,
                                           output_dtype=types.FLOAT,
                                           output_layout=types.NCHW,
                                           image_type=types.RGB,
                                           mean=mean,
                                           std=std)

        # To be able to handle all images from full-sized ImageNet, this padding sets the size of the internal
        # nvJPEG buffers without additional reallocations
        self.decode = ops.ImageDecoder(device=self.decode_device, output_type=types.RGB,
                                                 #device_memory_padding=211025920,
                                                 #host_memory_padding=140544512,
                                                 seed=seed)

        # Resize as desired.  To match torchvision data loader, use triangular interpolation.
        self.res = ops.Resize(device=self.dali_device, resize_x=size_x, resize_y=size_y,
                              interp_type=types.INTERP_TRIANGULAR)

        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        # Combined decode & random crop
        images = self.decode(self.jpegs)

        # Resize as desired
        images = self.res(images)

        output = self.cmn(images, mirror=rng)

        #self.labels = self.labels.gpu()
        return output, self.labels, rng


class HybridValPipe(Pipeline):
    """
    -resize to specified size
    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
        containing train & val subdirectories, with image class subfolders
    size (int): Resize size (typically 256 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    shard_id (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    """

    def __init__(self, batch_size, data_dir,file_list, num_threads, device_id, size_x, size_y,
                 mean, std, shard_id=0, shard_num=1, shuffle=False, seed=-1):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)

        # Enabling read_ahead slowed down processing ~40%
        # Note: initial_fill is for the shuffle buffer.  As we only want to see every example once, this is set to 1
        self.input = ops.FileReader(file_root=data_dir, file_list=file_list, shard_id=shard_id, num_shards=shard_num, random_shuffle=shuffle, initial_fill=1)
        self.decode_device = "mixed"
        self.dali_device = "gpu"

        self.cmnp = ops.CropMirrorNormalize(device=self.dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)

        self.decode = ops.ImageDecoder(device=self.decode_device, output_type=types.RGB)

        # Resize to desired size.  To match torchvision dataloader, use triangular interpolation
        self.res = ops.Resize(device=self.dali_device, resize_x=size_x, resize_y=size_y, interp_type=types.INTERP_TRIANGULAR)
        print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)

        return output, self.labels


class Kitti_Dataset_dali():

    def __init__(self, phase, data_root, cache_folder, conf, num_workers=1,  device_id=0):
        
        self.phase = phase
        self.data_root = data_root
        self.cache_folder = cache_folder
        self.fname = phase + '_imdb.pkl'
        self.file_list = os.path.join(cache_folder, phase+'_list.txt')
        seed = conf.cuda_seed
        self.image_mean = [i*255.0 for i in conf.image_means]
        self.image_std = [i*255.0 for i in conf.image_stds]

        if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, self.fname)):
            #logging.info('Preloading imdb.')
            self.imdb = pickle_read(os.path.join(cache_folder, self.fname))

        else:
            self.imdb = self.cache_data(conf)

        if not os.path.exists(self.file_list):
                self.cache_list(self.file_list)
        # store more information
        #self.datasets_train = conf.datasets_train
        #self.len = len(self.imdb)

        #if self.phase == 'train':
        #    # setup data augmentation transforms
        #    self.transform = 
        #        RandomMirror(self.mirror_prob),
        #        Resize(self.size),

        #elif self.phase =='test':
        #    self.transform = Preprocess(conf.test_scale, conf.image_means, conf.image_stds)

        self.check_cls(conf)

        # define iterator
        if phase == 'train':


            data_dir = os.path.join(self.data_root, conf.datasets_train[0]['name'], 'training', 'image_2_jpg')
            self.image_x = conf.crop_size[1]
            self.image_y = conf.crop_size[0]

            self.pipe = HybridTrainPipe(
                batch_size=conf.batch_size,
                data_dir = data_dir,
                file_list=self.file_list,
                num_threads=num_workers,
                device_id=device_id,
                size_y=conf.crop_size[0],
                size_x=conf.crop_size[1],
                mean=self.image_mean,
                std = self.image_std,
                shuffle=True,
                seed=conf.cuda_seed
            )

            self.pipe.build()
            self._iterator = DALIGenericIterator(self.pipe, ['images', 'indx', 'rng'], 
                            size=self.pipe.epoch_size("Reader"), fill_last_batch=True, last_batch_padded=False)

        elif phase == 'test':

            data_dir = os.path.join(self.data_root, conf.datasets_test[0]['name'], 'validation', 'image_2_jpg')
            self.image_x = conf.test_scale[1]
            self.image_y = conf.test_scale[0]

            self.pipe = HybridValPipe(
                batch_size=1,
                data_dir=data_dir,
                file_list=self.file_list,
                num_threads=num_workers,
                device_id=device_id,
                size_x=conf.test_scale[1],
                size_y=conf.test_scale[0],
                mean = self.image_mean,
                std = self.image_std,
                shuffle=True,
                seed = conf.cuda_seed
            )

            self.pipe.build()
            self._iterator = DALIGenericIterator(self.pipe, ['images', 'indx'],
                            size=self.pipe.epoch_size("Reader"), fill_last_batch=False, last_batch_padded=False)
        else:
            raise 'wrong phase'

        self.loader = self

    def __iter__(self):

        return self
    
    def __len__(self):

        return int(math.ceil(self._iterator._size / self._iterator.batch_size))

    def __next__(self):

        try:
            data = next(self._iterator)

        except StopIteration:
            #print('Resetting DALI loader')
            self._iterator.reset()
            raise StopIteration

        # Decode the data output
        images = data[0]['images']
        indxs = data[0]['indx']

        imobj = []
        if self.phase == 'train':
            rngs = data[0]['rng']
            for i in range(indxs.shape[0]):
                indx = indxs[i]
                objs = deepcopy(self.imdb[indx])
                rng = rngs[i]
                objs = self.label_mirror(rng, objs)
                objs = self.resize_label(objs)
                imobj.append(objs)        

        elif self.phase == 'test':
            assert indxs.shape[0] == 1, 'indxs.shape[0] != 1'
            indx = indxs[0]
            objs = deepcopy(self.imdb[indx])
            objs = self.resize_label(objs)
            imobj = objs        

        return images, imobj
        


    def check_cls(self, conf):
        # check classes
        cls_not_used = []

        for imobj in self.imdb:
            for gt in imobj.gts:

                cls = gt.cls
                if not(cls in conf.lbls or cls in conf.ilbls) and (cls not in cls_not_used):
                    cls_not_used.append(cls)

        if len(cls_not_used) > 0:
            logging.info('Labels not used in training.. {}'.format(cls_not_used))


    def cache_list(self, file_list):

        with open(file_list, 'w') as f:
            for indx, imobj in enumerate(self.imdb):
                f.write('{}.jpg {}\n'.format(imobj.id, int(imobj.id)))
        f.close()
            



    def cache_data(self, conf):

        for dbind, db in enumerate(conf['datasets_{}'.format(self.phase)]):

            logging.info('Loading imdb {}'.format(db['name']))

            # single imdb
            imdb_single_db = []

            # kitti formatting
            if db['anno_fmt'].lower() == 'kitti_det':
                if self.phase == 'train':
                    base_folder = os.path.join(self.data_root, db['name'], 'training')
                elif self.phase == 'val' or self.phase == 'test':
                    base_folder = os.path.join(self.data_root, db['name'], 'validation')

                ann_folder = os.path.join(base_folder, 'label_2', '')
                cal_folder = os.path.join(base_folder, 'calib', '')
                im_folder = os.path.join(base_folder, 'image_2', '')

                # get sorted filepaths
                annlist = sorted(glob(ann_folder + '*.txt'))

                imdb_start = time()

                self.affine_size = None if not ('affine_size' in conf) else conf.affine_size

                for annind, annpath in enumerate(annlist):

                    # get file parts
                    base = os.path.basename(annpath)
                    id, ext = os.path.splitext(base)

                    calpath = os.path.join(cal_folder, id + '.txt')
                    impath = os.path.join(im_folder, id + db['im_ext'])
                    impath_pre = os.path.join(base_folder, 'prev_2', id + '_01' + db['im_ext'])
                    impath_pre2 = os.path.join(base_folder, 'prev_2', id + '_02' + db['im_ext'])
                    impath_pre3 = os.path.join(base_folder, 'prev_2', id + '_03' + db['im_ext'])

                    # read gts
                    p2 = read_kitti_cal(calpath)
                    p2_inv = np.linalg.inv(p2)

                    gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)

                    if not self.affine_size is None:

                        # filter relevant classes
                        gts_plane = [deepcopy(gt) for gt in gts if gt.cls in conf.lbls and not gt.ign]

                        if len(gts_plane) > 0:

                            KITTI_H = 1.65

                            # compute ray traces for default projection
                            for gtind in range(len(gts_plane)):
                                gt = gts_plane[gtind]

                                #cx2d = gt.bbox_3d[0]
                                #cy2d = gt.bbox_3d[1]
                                cy2d = gt.bbox_full[1] + gt.bbox_full[3]
                                cx2d = gt.bbox_full[0] + gt.bbox_full[2] / 2

                                z2d, coord3d = projection_ray_trace(p2, p2_inv, cx2d, cy2d, KITTI_H)

                                gts_plane[gtind].center_in = coord3d[0:3, 0]
                                gts_plane[gtind].center_3d = np.array(gt.center_3d)


                            prelim_tra = np.array([gt.center_in for gtind, gt in enumerate(gts_plane)])
                            target_tra = np.array([gt.center_3d for gtind, gt in enumerate(gts_plane)])

                            if self.affine_size == 4:
                                prelim_tra = np.pad(prelim_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)
                                target_tra = np.pad(target_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)

                            affine_gt, err = solve_transform(prelim_tra, target_tra, compute_error=True)

                            a = 1

                    obj = edict()

                    # did not compute transformer
                    if (self.affine_size is None) or len(gts_plane) < 1:
                        obj.affine_gt = None
                    else:
                        obj.affine_gt = affine_gt

                    # store gts
                    obj.id = id
                    obj.gts = gts
                    obj.p2 = p2
                    obj.p2_inv = p2_inv

                    # im properties
                    #im = Image.open(impath)
                    im = cv2.imread(impath)
                    obj.path = impath
                    obj.imH, obj.imW, c = im.shape

                    # database properties
                    obj.dbname = db.name
                    obj.scale = db.scale
                    obj.dbind = dbind

                    # store
                    imdb_single_db.append(obj)

                    if (annind % 1000) == 0 and annind > 0:
                        time_str, dt = compute_eta(imdb_start, annind, len(annlist))
                        logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(annind, len(annlist), dt, time_str))


            # concatenate single imdb into full imdb
            imdb += imdb_single_db

        imdb = np.array(imdb)

        # cache off the imdb?
        if cache_folder is not None:
            pickle_write(os.path.join(cache_folder, self.fname), imdb)

        return imdb

    def label_mirror(self, rng, imobj):

        if rng:
            for gtind, gt in enumerate(imobj.gts):

                if 'bbox_full' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_full[0] = self.image_x - gt.bbox_full[0] - gt.bbox_full[2]

                if 'bbox_vis' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_vis[0] = self.image_x - gt.bbox_vis[0] - gt.bbox_vis[2]

                if 'bbox_3d' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_3d[0] = self.image_x - gt.bbox_3d[0] - 1
                    rotY = gt.bbox_3d[10]

                    rotY = (-math.pi - rotY) if rotY < 0 else (math.pi - rotY)

                    while rotY > math.pi: rotY -= math.pi * 2
                    while rotY < (-math.pi): rotY += math.pi * 2

                    cx2d = gt.bbox_3d[0]
                    cy2d = gt.bbox_3d[1]
                    cz2d = gt.bbox_3d[2]

                    coord3d = imobj.p2_inv.dot(np.array([cx2d * cz2d, cy2d * cz2d, cz2d, 1]))

                    alpha = convertRot2Alpha(rotY, coord3d[2], coord3d[0])

                    imobj.gts[gtind].bbox_3d[10] = rotY
                    imobj.gts[gtind].bbox_3d[6] = alpha
        
        return imobj

    def resize_label(self, imobj):

        scale_factor = self.image_y/imobj.imH
        imobj.scale_factor = scale_factor 

        # scale all coordinates
        for gtind, gt in enumerate(imobj.gts):

            if 'bbox_full' in imobj.gts[gtind]:
                #imobj.gts[gtind].bbox_full *= scale_factor
                imobj.gts[gtind].bbox_full *= scale_factor

            if 'bbox_vis' in imobj.gts[gtind]:
                imobj.gts[gtind].bbox_vis *= scale_factor

            if 'bbox_3d' in imobj.gts[gtind]:

                # only scale x/y center locations (in 2D space!)
                imobj.gts[gtind].bbox_3d[0] *= scale_factor
                imobj.gts[gtind].bbox_3d[1] *= scale_factor
        
        return imobj



        



        

class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIGenericIterator(pipelines=pipelines, size=size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target


def _preproc_worker(dali_iterator, cuda_stream, fp16, mean, std, output_queue, proc_next_input, done_event, pin_memory):
    """
    Worker function to parse DALI output & apply final pre-processing steps
    """

    while not done_event.is_set():
        # Wait until main thread signals to proc_next_input -- normally once it has taken the last processed input
        proc_next_input.wait()
        proc_next_input.clear()

        if done_event.is_set():
            print('Shutting down preproc thread')
            break

        try:
            data = next(dali_iterator)

            # Decode the data output
            input_orig = data[0]['data']
            target = data[0]['label'].squeeze().long()  # DALI should already output target on device

            # Copy to GPU and apply final processing in separate CUDA stream
            with torch.cuda.stream(cuda_stream):
                input = input_orig
                if pin_memory:
                    input = input.pin_memory()
                    del input_orig  # Save memory
                input = input.cuda(non_blocking=True)

                input = input.permute(0, 3, 1, 2)

                # Input tensor is kept as 8-bit integer for transfer to GPU, to save bandwidth
                if fp16:
                    input = input.half()
                else:
                    input = input.float()

                input = input.sub_(mean).div_(std)

            # Put the result on the queue
            output_queue.put((input, target))

        except StopIteration:
            print('Resetting DALI loader')
            dali_iterator.reset()
            output_queue.put(None)


class DaliIteratorCPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16=False, mean=(0., 0., 0.), std=(1., 1., 1.), pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')
        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.proc_next_input = Event()
        self.done_event = Event()
        self.output_queue = queue.Queue(maxsize=5)
        self.preproc_thread = threading.Thread(
            target=_preproc_worker,
            kwargs={'dali_iterator': self._dali_iterator, 'cuda_stream': self.stream, 'fp16': self.fp16, 'mean': self.mean, 'std': self.std, 'proc_next_input': self.proc_next_input, 'done_event': self.done_event, 'output_queue': self.output_queue, 'pin_memory': self.pin_memory})
        self.preproc_thread.daemon = True
        self.preproc_thread.start()

        self.proc_next_input.set()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.output_queue.get()
        self.proc_next_input.set()
        if data is None:
            raise StopIteration
        return data

    def __del__(self):
        self.done_event.set()
        self.proc_next_input.set()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preproc_thread.join()


class DaliIteratorCPUNoPrefetch(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    Note that permutation to channels first, converting from 8 bit to float & normalization are all performed on GPU
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    fp16 (bool): Use fp16 as output format, f32 otherwise
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    pin_memory (bool): Transfer input tensor to pinned memory, before moving to GPU
    """
    def __init__(self, fp16, mean, std, pin_memory=True, **kwargs):
        super().__init__(**kwargs)
        print('Using DALI CPU iterator')

        self.stream = torch.cuda.Stream()

        self.fp16 = fp16
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)
        self.pin_memory = pin_memory

        if self.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def __next__(self):
        data = next(self._dali_iterator)

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()  # DALI should already output target on device

        # Copy to GPU & apply final processing in seperate CUDA stream
        input = input.cuda(non_blocking=True)

        input = input.permute(0, 3, 1, 2)

        # Input tensor is transferred to GPU as 8 bit, to save bandwidth
        if self.fp16:
            input = input.half()
        else:
            input = input.float()

        input = input.sub_(self.mean).div_(self.std)
        return input, target
'''

class Kitti_Dataset_torch(torch.utils.data.Dataset):
    """
    A single Dataset class is used for the whole project,
    which implements the __init__ and __get__ functions from PyTorch.
    """

    def __init__(self, conf, paths=None, phase='train'):
        """
        This function reads in all datasets to be used in training and stores ANY relevant
        information which may be needed during training as a list of edict()
        (referred to commonly as 'imobj').

        The function also optionally stores the image database (imdb) file into a cache.
        """

        # add for transform targets
        self.num_classes = len(conf.lbls) + 1
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
        self.feat_stride = conf.feat_stride
        self.feat_size = [int(i/conf.feat_stride) for i in conf.crop_size]

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
        #########################################

        imdb = []

        self.use_3d_for_2d = ('use_3d_for_2d' in conf) and conf.use_3d_for_2d
        self.fname = phase+'_imdb.pkl'
        if phase == 'val_train':
            self.fname = 'train_imdb.pkl'
        self.phase = phase
        # use cache?
        cache_folder = paths.data_cache
        root = paths.data
        self.cache_folder = cache_folder
        self.root = root
        if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, self.fname)):
            logging.info('Preloading imdb.')
            imdb = pickle_read(os.path.join(cache_folder, self.fname))

        else:

            # cycle through each dataset
            for dbind, db in enumerate(conf['datasets_{}'.format(self.phase)]):

                logging.info('Loading imdb {}'.format(db['name']))

                # single imdb
                imdb_single_db = []

                # kitti formatting
                if db['anno_fmt'].lower() == 'kitti_det':
                    if self.phase == 'train':
                        base_folder = os.path.join(root, db['name'], 'training')
                        ann_folder = os.path.join(base_folder, 'label_2', '')
                    elif self.phase == 'validation':
                        base_folder = os.path.join(root, db['name'], 'validation')
                        ann_folder = None
                    elif self.phase == 'test':
                        base_folder = os.path.join(root, db['name'], 'testing')
                        ann_folder = None


                    #ann_folder = os.path.join(base_folder, 'label_2', '')
                    cal_folder = os.path.join(base_folder, 'calib', '')
                    im_folder = os.path.join(base_folder, 'image_2', '')

                    # get sorted filepaths
                    #annlist = sorted(glob(ann_folder + '*.txt'))
                    imlist = sorted(glob(im_folder + '*.png'))

                    imdb_start = time()

                    self.affine_size = None if not ('affine_size' in conf) else conf.affine_size

                    #for annind, annpath in enumerate(annlist):
                    for imind, impath in enumerate(imlist):

                        # get file parts
                        #base = os.path.basename(annpath)
                        base = os.path.basename(impath)
                        id, ext = os.path.splitext(base)

                        calpath = os.path.join(cal_folder, id + '.txt')
                        # read gts
                        p2 = read_kitti_cal(calpath)
                        p2_inv = np.linalg.inv(p2)

                        #impath = os.path.join(im_folder, id + db['im_ext'])
                        if self.phase == 'train':
                            annpath = os.path.join(ann_folder, id + '.txt')
                            gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)
                        else: 
                            gts = None


                        '''
                        if not self.affine_size is None:

                            # filter relevant classes
                            gts_plane = [deepcopy(gt) for gt in gts if gt.cls in conf.lbls and not gt.ign]

                            if len(gts_plane) > 0:

                                KITTI_H = 1.65

                                # compute ray traces for default projection
                                for gtind in range(len(gts_plane)):
                                    gt = gts_plane[gtind]

                                    #cx2d = gt.bbox_3d[0]
                                    #cy2d = gt.bbox_3d[1]
                                    cy2d = gt.bbox_full[1] + gt.bbox_full[3]
                                    cx2d = gt.bbox_full[0] + gt.bbox_full[2] / 2

                                    z2d, coord3d = projection_ray_trace(p2, p2_inv, cx2d, cy2d, KITTI_H)

                                    gts_plane[gtind].center_in = coord3d[0:3, 0]
                                    gts_plane[gtind].center_3d = np.array(gt.center_3d)


                                prelim_tra = np.array([gt.center_in for gtind, gt in enumerate(gts_plane)])
                                target_tra = np.array([gt.center_3d for gtind, gt in enumerate(gts_plane)])

                                if self.affine_size == 4:
                                    prelim_tra = np.pad(prelim_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)
                                    target_tra = np.pad(target_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)

                                affine_gt, err = solve_transform(prelim_tra, target_tra, compute_error=True)

                                a = 1
                        '''

                        obj = edict()

                        # did not compute transformer
                        '''
                        if (self.affine_size is None) or len(gts_plane) < 1:
                            obj.affine_gt = None
                        else:
                            obj.affine_gt = affine_gt
                        '''

                        # store gts
                        obj.id = id
                        obj.gts = gts
                        obj.p2 = p2
                        obj.p2_inv = p2_inv

                        # im properties
                        #im = Image.open(impath)
                        im = cv2.imread(impath)
                        obj.path = impath
                        obj.imH, obj.imW, c = im.shape

                        # database properties
                        obj.dbname = db.name
                        obj.scale = db.scale
                        obj.dbind = dbind

                        # store
                        imdb_single_db.append(obj)
                        if (imind % 1000) == 0 and imind > 0:
                            time_str, dt = compute_eta(imdb_start, imind, len(imlist))
                            logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(imind, len(imlist), dt, time_str))


                # concatenate single imdb into full imdb
                imdb += imdb_single_db

            imdb = np.array(imdb)

            # cache off the imdb?
            if cache_folder is not None:
                pickle_write(os.path.join(cache_folder, self.fname), imdb)

        # store more information
        #self.datasets_train = conf.datasets_train
        self.len = len(imdb)
        self.imdb = imdb

        if conf.anchors is None and self.phase == 'train':
            generate_anchors(conf, self.imdb, paths.data_cache) # generate anchors to conf.anchors
            compute_bbox_stats(conf, self.imdb, paths.data_cache)

            self.num_anchors = conf.anchors.shape[0]
            self.anchors = conf.anchors
            self.bbox_means = conf.bbox_means
            self.bbox_stds = conf.bbox_stds

        if self.phase == 'train':
            # setup data augmentation transforms
            self.transform = Augmentation(conf)

            # setup sampler and data loader for this dataset
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(balance_samples(conf, imdb), self.len)
            #self.loader = torch.utils.data.DataLoader(self, conf.batch_size, sampler=self.sampler, collate_fn=self.collate, num_workers=16, pin_memory=True)
            self.loader = torch.utils.data.DataLoader(self, conf.batch_size, sampler=self.sampler, num_workers=conf.num_workers, pin_memory=True)
            #self.loader = torch.utils.data.DataLoader(self, conf.batch_size, num_workers=2*conf.batch_size, pin_memory=True, shuffle=True, drop_last=True)
        elif self.phase =='test' or self.phase == 'validation':
            self.transform = Preprocess(conf.test_scale, conf.image_means, conf.image_stds)
            self.sampler = None
            self.loader = torch.utils.data.DataLoader(self, 1, num_workers=1)


        # check classes
        if self.phase == 'train':
            cls_not_used = []
            for imobj in imdb:

                for gt in imobj.gts:
                    cls = gt.cls
                    if not(cls in conf.lbls or cls in conf.ilbls) and (cls not in cls_not_used):
                        cls_not_used.append(cls)

            if len(cls_not_used) > 0:
                logging.info('Labels not used in training.. {}'.format(cls_not_used))

        self.ind_tmp = 1

    def __getitem__(self, index):
        """
        Grabs the item at the given index. Specifically,
          - read the image from disk
          - read the imobj from RAM
          - applies data augmentation to (im, imobj)
          - converts image to RGB and [B C W H]
        """
        #print(index)
        # read images
        im = cv2.imread(self.imdb[index].path)
        #cv2.imwrite('./atten_weights/{}.png'.format(self.ind_tmp), im)
        #self.ind_tmp += 1

        # transform / data augmentation
        im, imobj = self.transform(im, deepcopy(self.imdb[index]))


        if im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            for i in range(int(im.shape[2]/3)):
                # convert to RGB then permute to be [B C H W]
                im[:, :, (i*3):(i*3) + 3] = im[:, :, (i*3+2, i*3+1, i*3)]

        #im = np.transpose(im, [2, 0, 1])
        im = torch.from_numpy(im).permute(2, 0, 1)

        if self.phase == 'train':
            labels_fg, labels_bg, labels_ign, labels, bbox_2d, bbox_3d, rois, any_val = self._targets(imobj)
        elif self.phase == 'test' or self.phase == 'validation':
            labels_fg, labels_bg, labels_ign, labels, bbox_2d, bbox_3d, rois, any_val = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            raise 'wrong phase'

        meta = {
            'p2': imobj.p2,
            'p2_inv': imobj.p2_inv,
            'imH': imobj.imH,
            'imW': imobj.imW,
            'scale_factor':imobj.scale_factor,
            'rois':rois,
            'id':imobj.id,
            'any_val': any_val
        }
        target = {
            'labels_fg': labels_fg,
            'labels_bg': labels_bg,
            'labels_ign': labels_ign,
            'labels' : labels,
            'bbox_2d': bbox_2d,
            'bbox_3d': bbox_3d,
            'meta'  : meta
        }
        ret = {
            'input': im,
            'target': target,
        }
        return ret

    #@staticmethod
    #def collate(batch):
    #    """
    #    Defines the methodology for PyTorch to collate the objects
    #    of a batch together, for some reason PyTorch doesn't function
    #    this way by default.
    #    """

    #    imgs = []
    #    imobjs = []

    #    # go through each batch
    #    for sample in batch:
    #        
    #        # append images and object dictionaries
    #        imgs.append(sample[0])
    #        imobjs.append(sample[1])

    #    # stack images
    #    #imgs = np.array(imgs)
    #    imgs = torch.stack(imgs, dim=0)

    #    return imgs, imobjs

    def __len__(self):
        """
        Simply return the length of the dataset.
        """
        return self.len

    def _targets(self, imobjs):

        # origin
        #def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size):

        total_anchors = self.num_anchors * self.feat_size[0] * self.feat_size[1]
        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        labels = np.zeros(total_anchors, dtype=int)

        # x, y, w, h
        bbox_2d = np.zeros([total_anchors, 4], dtype=np.float32)

        # x, y, z, w, h, l, ry
        bbox_3d = np.zeros([total_anchors, 7], dtype=np.float32)
        #bbox_x3d_tar = np.zeros(total_anchors)
        #bbox_y3d_tar = np.zeros(total_anchors)
        #bbox_z3d_tar = np.zeros(total_anchors)
        #bbox_w3d_tar = np.zeros(total_anchors)
        #bbox_h3d_tar = np.zeros(total_anchors)
        #bbox_l3d_tar = np.zeros(total_anchors)
        #bbox_ry3d_tar = np.zeros(total_anchors)

        # get all rois
        rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride)
        rois = rois.astype(np.float32)

        #src_anchors = self.anchors[rois[:, 4].type(torch.cuda.LongTensor), :]
        src_anchors = self.anchors[rois[:, 4].astype(int), :]
        #if len(src_anchors.shape) == 1: src_anchors = src_anchors[np.newaxis, :]
        if len(src_anchors.shape) == 1: raise 'src_anchors: {}'.format(src_anchors)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        #for bind in range(0, batch_size):

        if type(imobjs) is dict:
            imobjs = edict(imobjs)
        gts = imobjs.gts

        p2_inv = imobjs.p2_inv

        # filter gts
        igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

        # accumulate boxes
        gts_all = bbXYWH2Coords(np.array([gt.bbox_full for gt in gts]))
        gts_3d = np.array([gt.bbox_3d for gt in gts])

        #if not ((rmvs == False) & (igns == False)).any():
        #    continue

        # filter out irrelevant cls, and ignore cls
        gts_val = gts_all[(rmvs == False) & (igns == False), :]
        gts_ign = gts_all[(rmvs == False) & (igns == True), :]
        gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

        # accumulate labels
        box_lbls = np.array([gt.cls for gt in gts])
        box_lbls = box_lbls[(rmvs == False) & (igns == False)]
        box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

        if gts_val.shape[0] > 0:

            # bbox regression
            transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois, self.fg_thresh,
                                              self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                              self.best_thresh, anchors=self.anchors,  gts_3d=gts_3d,
                                              tracker=rois[:, 4])

            # normalize 2d
            transforms[:, 0:4] -= self.bbox_means[:, 0:4]
            transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

            # normalize 3d
            transforms[:, 5:12] -= self.bbox_means[:, 4:]
            transforms[:, 5:12] /= self.bbox_stds[:, 4:]

            labels_fg  = (transforms[:, 4] > 0).astype(int)
            labels_bg  = (transforms[:, 4] < 0).astype(int)
            labels_ign = (transforms[:, 4] == 0).astype(int)

            fg_inds = np.flatnonzero(labels_fg)
            bg_inds = np.flatnonzero(labels_bg)
            ign_inds = np.flatnonzero(labels_ign)


            labels[fg_inds] = transforms[fg_inds, 4]
            labels[ign_inds] = IGN_FLAG
            labels[bg_inds] = 0

            bbox_2d[:, :] = transforms[:, 0:4]
            #bbox_x_tar = transforms[:, 0]
            #bbox_y_tar = transforms[:, 1]
            #bbox_w_tar = transforms[:, 2]
            #bbox_h_tar = transforms[:, 3]

            # x, y, z, w, h, l, ry
            bbox_3d[:, :] = transforms[:, 5:12]

            #bbox_x3d_tar = transforms[:, 5]
            #bbox_y3d_tar = transforms[:, 6]
            #bbox_z3d_tar = transforms[:, 7]
            #bbox_w3d_tar = transforms[:, 8]
            #bbox_h3d_tar = transforms[:, 9]
            #bbox_l3d_tar = transforms[:, 10]
            #bbox_ry3d_tar = transforms[:, 11]

        else:

            labels_fg  = np.zeros(total_anchors, dtype=int)
            labels_bg  = np.ones(total_anchors, dtype=int)
            labels_ign = np.zeros(total_anchors, dtype=int)

        any_val = int(((rmvs == False) & (igns == False)).any())
        #res = {

        #    'bg_inds': bg_inds,
        #    'fg_inds': fg_inds,
        #    'labels' : labels,
        #    'box_2d' : bbox_2d,
        #    'box_3d' : bbox_3d,
        #}
        return  labels_fg, labels_bg, labels_ign, labels, bbox_2d, bbox_3d, rois, any_val

        # grab label predictions (for weighing purposes)


if __name__ == '__main__':

    import os
    import sys
    cwd = os.getcwd()
    sys.path.append(os.path.join(cwd, '../'))

    from scripts.config.kitti_3d_multi_main import Config
    from tqdm import tqdm, trange

    data_root = '/home/luoshujie/Project/M3D_RPN/data'
    cache_folder = '/home/luoshujie/Project/M3D_RPN/data_cache'
    file_list = 'file_list.txt'
    phase = 'train'
    conf = Config()
    cache_folder = os.path.join(cache_folder, conf.datasets_train[0]['name'])

    dataloader = Kitti_Dataset(phase, data_root, cache_folder, conf, num_workers=4)

    epoch = 50
    for i in trange(epoch):
        for data, imobj in tqdm(dataloader):
            
            print(torch.max(data))
            print(torch.min(data))
            print(torch.mean(data))
            print(torch.std(data))
