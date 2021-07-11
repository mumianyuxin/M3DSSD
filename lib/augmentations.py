"""
This file contains all PyTroch data augmentation functions.

Every transform should have a __call__ function which takes in (self, image, imobj)
where imobj is an arbitary dict containing relevant information to the image.

In many cases the imobj can be None, which enables the same augmentations to be used
during testing as they are in training.

Optionally, most transforms should have an __init__ function as well, if needed.
"""

import numpy as np
from numpy import random
import cv2
import math
import os
import sys

from lib.util import *


class Compose(object):
    """
    Composes a set of functions which take in an image and an object, into a single transform
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, imobj=None):
        for t in self.transforms:
            img, imobj = t(img, imobj)
        return img, imobj


class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, image, imobj=None):
        return image.astype(np.float32), imobj


class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, image, imobj=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= np.tile(self.mean, int(image.shape[2]/self.mean.shape[0]))
        image /= np.tile(self.stds, int(image.shape[2]/self.stds.shape[0]))
        return image.astype(np.float32), imobj


class Resize(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image, imobj=None):

        scale_factor = self.size[0] / image.shape[0]

        #h = np.round(image.shape[0] * scale_factor).astype(int)
        #w = np.round(image.shape[1] * scale_factor).astype(int)
        h = self.size[0]
        w = self.size[1]

        # resize
        image = cv2.resize(image, (w, h))

        if len(self.size) > 1:

            # crop in
            if image.shape[1] > self.size[1]:
                image = image[:, 0:self.size[1], :]

            # pad out
            elif image.shape[1] < self.size[1]:
                padW = self.size[1] - image.shape[1]
                image = np.pad(image, [(0, 0), (0, padW), (0, 0)], 'constant')

        if imobj:

            # store scale factor, just in case
            imobj.scale_factor = scale_factor

            if 'gts' in imobj:

                # scale all coordinates
                for gtind, gt in enumerate(imobj.gts):

                    if 'bbox_full' in imobj.gts[gtind]:
                        imobj.gts[gtind].bbox_full *= scale_factor

                    if 'bbox_vis' in imobj.gts[gtind]:
                        imobj.gts[gtind].bbox_vis *= scale_factor

                    if 'bbox_3d' in imobj.gts[gtind]:

                        # only scale x/y center locations (in 2D space!)
                        imobj.gts[gtind].bbox_3d[0] *= scale_factor
                        imobj.gts[gtind].bbox_3d[1] *= scale_factor

            if 'gts_pre' in imobj:

                # scale all coordinates
                for gtind, gt in enumerate(imobj.gts_pre):

                    if 'bbox_full' in imobj.gts_pre[gtind]:
                        imobj.gts_pre[gtind].bbox_full *= scale_factor

                    if 'bbox_vis' in imobj.gts_pre[gtind]:
                        imobj.gts_pre[gtind].bbox_vis *= scale_factor

                    if 'bbox_3d' in imobj.gts_pre[gtind]:

                        # only scale x/y center locations (in 2D space!)
                        imobj.gts_pre[gtind].bbox_3d[0] *= scale_factor
                        imobj.gts_pre[gtind].bbox_3d[1] *= scale_factor

        return image, imobj

class Padding(object):
    """
    Resize the image according to the target size height and the image height.
    If the image needs to be cropped after the resize, we crop it to self.size,
    otherwise we pad it with zeros along the right edge

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, size):
        self.size = size
        self.value = [0, 0, 0]

    def __call__(self, image, imobj=None):

        h, w, _ = image.shape

        image = cv2.copyMakeBorder(image, 0, self.size[0]-h, 0, self.size[1]-w, cv2.BORDER_CONSTANT, value=self.value)

        #trans_mat = cv2.getRotationMatrix2D((0.5*w, 0.5*h), 0, 1.0)
        #image = cv2.warpAffine(image, trans_mat, (self.size[1], self.size[0]))

        if imobj:
            # store scale factor, just in case
            imobj.scale_factor = 1.0

        return image, imobj



class RandomTransform(object):
    """
    transform the image including scaling, shifting.

    If the object has ground truths we also scale the (known) box coordinates.
    """
    def __init__(self, distort_prob=0.7, shift=0.2, scale=0.4, dst_h=384, dst_w=1280):
        '''
        parameter:
          dst_shape = (w, h)
        '''
        self.scale = scale
        self.shift = shift
        self.dst_shape = (dst_w, dst_h)
        self.distort_prob = distort_prob

    def __call__(self, im, imobj=None):

        if random.rand() < self.distort_prob:
            scale = np.clip(np.random.randn()*self.scale, - self.scale, self.scale) + 1
            center_x = im.shape[1] * (0.5 + np.clip(np.random.randn()*self.shift, -2*self.shift, 2*self.shift))
            center_y = im.shape[0] * (0.5 + np.clip(np.random.randn()*self.shift, -2*self.shift, 2*self.shift))
            aug = True

        else:
            scale = 1.0
            center_x = im.shape[1] * 0.5
            center_y = im.shape[0] * 0.5
            aug = False

        trans_mat = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

        # tranform img
        im = cv2.warpAffine(im, trans_mat, self.dst_shape)

        # transfomr imobj
        if imobj:
                        # store scale factor, just in case
            imobj.scale_factor = scale

            if 'gts' in imobj and aug:

                # scale all coordinates
                for gtind, gt in enumerate(imobj.gts):

                    if 'bbox_full' in imobj.gts[gtind]:
                        imobj.gts[gtind].bbox_full[2:4] *= scale
                        imobj.gts[gtind].bbox_full[0:2] = affine_transform(gt.bbox_full[0:2], trans_mat)

                    if 'bbox_vis' in imobj.gts[gtind]:
                        imobj.gts[gtind].bbox_vis *= scale

                    if 'bbox_3d' in imobj.gts[gtind]:
                        # transform 2d coords
                        cx, cy = affine_transform(gt.bbox_3d[0:2], trans_mat)

                        # transform 3d coords
                        cz3d_2d = imobj.gts[gtind].bbox_3d[2] / scale
                        
                        imobj.gts[gtind].bbox_3d[0:3] = [cx, cy, cz3d_2d]
                        #z_3d = imobj.gts[gtind].bbox_3d[9]
                        cx3d, cy3d, cz3d, _ = imobj.p2_inv.dot(np.array([cx* cz3d_2d, cy * cz3d_2d, 1 * cz3d_2d, 1]))
                        #print('err:', cz3d-imobj.gts[gtind].center_3d[2]/scale)
                        imobj.gts[gtind].center_3d = [cx3d, cy3d, cz3d]
                        imobj.gts[gtind].bbox_3d[7:10] = [cx3d, cy3d, cz3d]

                        alpha = imobj.gts[gtind].bbox_3d[6]
                        rotY = convertAlpha2Rot(alpha, cz3d, cx3d)
                        imobj.gts[gtind].bbox_3d[10] = rotY
        
        return im, imobj

class RandomSaturation(object):
    """
    Randomly adjust the saturation of an image given a lower and upper bound,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.distort_prob = distort_prob
        self.lower = lower
        self.upper = upper

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, imobj=None):
        if random.rand() <= self.distort_prob:
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, imobj


class RandomHue(object):
    """
    Randomly adjust the hue of an image given a delta degree to rotate by,
    and a distortion probability.

    This function assumes the image is in HSV!!
    """
    def __init__(self, distort_prob, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, image, imobj=None):
        if random.rand() <= self.distort_prob:
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, imobj


class ConvertColor(object):
    """
    Converts color spaces to/from HSV and BGR
    """
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, imobj=None):

        # BGR --> HSV
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV --> BGR
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        else:
            raise NotImplementedError

        return image, imobj


class RandomContrast(object):
    """
    Randomly adjust contrast of an image given lower and upper bound,
    and a distortion probability.
    """
    def __init__(self, distort_prob, lower=0.5, upper=1.5):

        self.lower = lower
        self.upper = upper
        self.distort_prob = distort_prob

        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, imobj=None):
        if random.rand() <= self.distort_prob:
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, imobj


class RandomMirror(object):
    """
    Randomly mirror an image horzontially, given a mirror probabilty.

    Also, adjust all box cordinates accordingly.
    """
    def __init__(self, mirror_prob):
        self.mirror_prob = mirror_prob

    def __call__(self, image, imobj):

        _, width, _ = image.shape

        if random.rand() <= self.mirror_prob:

            image = image[:, ::-1, :]
            image = np.ascontiguousarray(image)

            # flip the coordinates w.r.t the horizontal flip (only adjust X)
            for gtind, gt in enumerate(imobj.gts):

                if 'bbox_full' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_full[0] = image.shape[1] - gt.bbox_full[0] - gt.bbox_full[2]

                if 'bbox_vis' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_vis[0] = image.shape[1] - gt.bbox_vis[0] - gt.bbox_vis[2]

                if 'bbox_3d' in imobj.gts[gtind]:
                    imobj.gts[gtind].bbox_3d[0] = image.shape[1] - gt.bbox_3d[0] - 1
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


        return image, imobj


class RandomBrightness(object):
    """
    Randomly adjust the brightness of an image given given a +- delta range,
    and a distortion probability.
    """
    def __init__(self, distort_prob, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.distort_prob = distort_prob

    def __call__(self, image, imobj=None):
        if random.rand() <= self.distort_prob:
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, imobj


class PhotometricDistort(object):
    """
    Packages all photometric distortions into a single transform.
    """
    def __init__(self, distort_prob):

        self.distort_prob = distort_prob

        # contrast is duplicated because it may happen before or after
        # the other transforms with equal probability.
        self.transforms = [
            RandomContrast(distort_prob),
            ConvertColor(transform='HSV'),
            RandomSaturation(distort_prob),
            RandomHue(distort_prob),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast(distort_prob)
        ]

        self.rand_brightness = RandomBrightness(distort_prob)

    def __call__(self, image, imobj):

        # do contrast first
        if random.rand() <= 0.5:
            distortion = self.transforms[:-1]

        # do contrast last
        else:
            distortion = self.transforms[1:]

        # add random brightness
        distortion.insert(0, self.rand_brightness)

        # compose transformation
        distortion = Compose(distortion)

        return distortion(image.copy(), imobj)


class Augmentation(object):
    """
    Data Augmentation class which packages the typical pre-processing
    and all data augmentation transformations (mirror and photometric distort)
    into a single transform.
    """
    def __init__(self, conf):

        self.mean = conf.image_means
        self.stds = conf.image_stds
        self.size = conf.crop_size
        self.mirror_prob = conf.mirror_prob
        self.distort_prob = conf.distort_prob
        self.trans_prob = conf.trans_prob
        self.shift = conf.shift
        self.scale_trans = conf.scale_trans

        if conf.distort_prob <= 0:
            self.augment = Compose([
                ConvertToFloat(),
                RandomMirror(self.mirror_prob),
                RandomTransform(self.trans_prob, conf.shift, conf.scale_trans, dst_h=self.size[0], dst_w=self.size[1]),
                #Resize(self.size),
                Normalize(self.mean, self.stds)
            ])
        else:
            self.augment = Compose([
                ConvertToFloat(),
                PhotometricDistort(self.distort_prob),
                RandomMirror(self.mirror_prob),
                RandomTransform(self.trans_prob, conf.shift, conf.scale_trans, dst_h=self.size[0], dst_w=self.size[1]),
                #Resize(self.size),
                Normalize(self.mean, self.stds)
            ])

    def __call__(self, img, imobj):
        return self.augment(img, imobj)


class Preprocess(object):
    """
    Preprocess function which ONLY does the basic pre-processing of an image,
    meant to be used during the testing/eval stages.
    """
    def __init__(self, size, mean, stds):

        self.mean = mean
        self.stds = stds
        self.size = size

        self.preprocess = Compose([
            ConvertToFloat(),
            Padding(self.size),
            #Resize(self.size),
            Normalize(self.mean, self.stds)
        ])

    def __call__(self, img, imobj=None):

        return self.preprocess(img, imobj)

        #for i in range(int(img.shape[2]/3)):

        #    # convert to RGB then permute to be [B C H W]
        #    img[:, :, (i*3):(i*3) + 3] = img[:, :, (i*3+2, i*3+1, i*3)]

        #img = np.transpose(img, [2, 0, 1])

        #return img
