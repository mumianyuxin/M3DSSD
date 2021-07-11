# M3DSSD: Monocular 3D Single Stage Object Detector

### Setup

- pytorch 0.4.1

- Preparation

  Download the full [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) detection dataset. Then place a softlink (or the actual data) in M3DSSD/data/kitti*.

  ```shell
   cd M3DSSD
   ln -s /path/to/kitti data/kitti
  ```

  Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage.

  ```sh
  # extract the data splits
  python data/kitti_split1/setup_split.py
  
  # build  the KITTI devkit eval for each split.
  sh data/kitti_split1/devkit/cpp/build.sh
  ```

  Build the nms modules

  ```
  cd lib/nms
  make
  ```

  Build the DCN modules

  ```
  cd model/DCNv2
  sh ./make.sh
  ```

  

- Training

  Review the configurations in *scripts/config* for details.

  ```
  python scripts/train_rpn_3d.py --config=kitti_3d_base --exp_name base
  ```
  - Tips: It is recommended to load a pre-trained model when training with feature alignment.

- Testing
  
  Modify the `conf_path` and `weights_path` to run test. 
  ```
  python scripts/test_rpn_3d.py
  ```
  
## Acknowledgements
- Thanks [Garrick Brazil](https://github.com/garrickbrazil/M3D-RPN) for his great works.
- Thanks [CharlesShang](https://github.com/CharlesShang/DCNv2) for his works.
- Thanks [traveller59](https://github.com/traveller59/kitti-object-eval-python) for his works
