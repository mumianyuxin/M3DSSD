from easydict import EasyDict as edict
import numpy as np

def Config():

    conf = edict()
        
    # ----------------------------------------
    #  general
    # ----------------------------------------

    conf.model = 'M3d_inference_align'
    conf.ida_dcnv2 = True
    conf.attention = "ANAB"

    # align 
    conf.center_align = False
    conf.shape_align = False
    
    # solver settings
    conf.solver_type = 'sgd'
    conf.lr = 0.002
    conf.momentum = 0.9
    conf.weight_decay = 0.0005
    conf.max_epoch = 70
    conf.warmup = 1.0/conf.max_epoch
    conf.eval_epoch = 20
    conf.snapshot_epoch = 5
    conf.display_iter = 50
    conf.do_test = True
    
    # sgd parameters
    conf.lr_policy = 'cos'
    conf.lr_steps = None
    conf.lr_target = conf.lr * 1e-5
    
    # random
    conf.rng_seed = 2
    conf.cuda_seed = 2
    
    # misc network
    conf.image_means = [0.485, 0.456, 0.406]
    conf.image_stds = [0.229, 0.224, 0.225]
    conf.feat_stride = 8
    conf.back_bone = 'dla102'
    conf.pre_train = True
    
    conf.has_3d = True

    # ----------------------------------------
    #  image sampling and datasets
    # ----------------------------------------

    # scale sampling  
    conf.test_scale = [384, 1280]
    conf.crop_size = [384, 1280]
    conf.mirror_prob = 0.50
    conf.trans_prob = 0.7
    conf.distort_prob = -1
    conf.shift = 0.1
    conf.scale_trans = 0.4
    
    # datasets
    #conf.dataset_test = 'kitti_split1'
    conf.datasets_validation = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.datasets_test = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.datasets_train = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.use_3d_for_2d = True
    conf.mc = False
    conf.num_workers = 8
    
    # percent expected height ranges based on test_scale
    # used for anchor selection 
    conf.percent_anc_h = [0.0625, 0.75]
    
    # labels settings
    conf.min_gt_h = conf.test_scale[0]*conf.percent_anc_h[0]
    conf.max_gt_h = conf.test_scale[0]*conf.percent_anc_h[1]
    conf.min_gt_vis = 0.65
    conf.ilbls = ['Van', 'ignore']
    conf.lbls = ['Car', 'Pedestrian', 'Cyclist']
    
    # ----------------------------------------
    #  detection sampling
    # ----------------------------------------
    
    # detection sampling
    conf.batch_size = 4
    conf.fg_image_ratio = 1.0
    conf.box_samples = 0.20
    conf.fg_fraction = 0.20
    conf.bg_thresh_lo = 0
    conf.bg_thresh_hi = 0.5
    conf.fg_thresh = 0.5
    conf.ign_thresh = 0.5
    conf.best_thresh = 0.35

    # ----------------------------------------
    #  inference and testing
    # ----------------------------------------

    # nms
    conf.nms_topN_pre = 3000
    conf.nms_topN_post = 40
    conf.nms_thres = 0.4
    conf.clip_boxes = False

    conf.test_protocol = 'kitti'
    conf.test_db = 'kitti'
    conf.test_min_h = 0
    conf.min_det_scales = [0, 0]

    # ----------------------------------------
    #  anchor settings
    # ----------------------------------------
    
    # clustering settings
    conf.cluster_anchors = 0
    conf.even_anchors = 0
    conf.expand_anchors = 0
                             
    conf.anchors = None

    conf.bbox_means = None
    conf.bbox_stds = None
    
    # initialize anchors
    base = (conf.max_gt_h / conf.min_gt_h) ** (1 / (12 - 1))
    conf.anchor_scales = np.array([conf.min_gt_h * (base ** i) for i in range(0, 12)])
    conf.anchor_ratios = np.array([0.5, 1.0, 1.5])
    
    # loss logic
    conf.hard_negatives = True
    conf.focal_loss = 0
    conf.cls_2d_lambda = 1
    conf.iou_2d_lambda = 1
    conf.bbox_2d_lambda = 0
    conf.bbox_3d_lambda = 1
    conf.bbox_3d_proj_lambda = 0.0
    conf.bbox_3d_iou_lambda = 0
    conf.pre_compute_target = True
    
    conf.hill_climbing = True
    
    conf.bins = 32
    
    
    #conf.pretrained = 'output/3d_iou/kitti_3d_multi_warmup/weights/model_' + conf.back_bone + '_best_pkl'

    return conf
