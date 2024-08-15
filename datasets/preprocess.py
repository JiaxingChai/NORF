import os
import cv2
import torch
import warnings
import random
import json
from PIL import ImageEnhance
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor as torchtotensor

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")


class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label, border=None):
        if border is None:
            for t in self.transforms:
                img, label = t(img, label)
            return img, label
        else:
            for t in self.transforms:
                img, label, border = t(img, label, border)
            return img, label, border

class Random_crop_Resize_Video(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels, borders=None):
        res_img = []
        res_label = []
        res_border = []
        
        if borders is None:
            for img, label in zip(imgs, labels):
                x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
                width, height = img.size
                region = [x, y, width - x, height - y]
                
                img = img.crop(region).resize((width, height), Image.BILINEAR)
                label = label.crop(region).resize((width, height), Image.NEAREST)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            for img, label, border in zip(imgs, labels, borders):
                x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
                width, height = img.size
                region = [x, y, width - x, height - y]
                
                img = img.crop(region).resize((width, height), Image.BILINEAR)
                label = label.crop(region).resize((width, height), Image.NEAREST)
                border = border.crop(region).resize((img.size[0], img.size[1]), Image.NEAREST)
                res_img.append(img)
                res_label.append(label)
                res_border.append(border)
            return res_img, res_label, res_border

class Random_horizontal_flip_video(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels, borders=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if borders is None:
            if random.random() < self.prob:
                res_img = []
                res_label = []
                for img, label in zip(imgs, labels):
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    label = label.transpose(Image.FLIP_LEFT_RIGHT)
                    res_img.append(img)
                    res_label.append(label)
                return res_img, res_label
            else:
                return imgs, labels
        else:
            if random.random() < self.prob:
                res_img = []
                res_label = []
                res_border = []
                for img, label, border in zip(imgs, labels, borders):
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    label = label.transpose(Image.FLIP_LEFT_RIGHT)
                    border = border.transpose(Image.FLIP_LEFT_RIGHT)
                    res_img.append(img)
                    res_label.append(label)
                    res_border.append(border)
                return res_img, res_label, res_border
            else:
                return imgs, labels, borders

class Resize_video(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels, borders=None):
        res_img = []
        res_label = []
        res_border = []
        
        if borders is None:
            for img, label in zip(imgs, labels):
                res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
                res_label.append(label.resize((self.width, self.height), Image.NEAREST))
            return res_img, res_label
        else:
            for img, label, border in zip(imgs, labels, borders):
                res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
                res_label.append(label.resize((self.width, self.height), Image.NEAREST))
                res_border.append(border.resize((self.width, self.height), Image.NEAREST))
            return res_img, res_label, res_border

class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels, borders=None):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[i, :, :] -= float(self.mean[i])
            for i in range(3):
                img[i, :, :] /= float(self.std[i])
            res_img.append(img)
        
        if borders is None:
            return res_img, labels
        else:  
            return res_img, labels, borders

class toTensor_video(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels, borders=None):
        res_img = []
        res_label = []
        res_border = []
        
        if borders is None:
            for img, label in zip(imgs, labels):
                img = self.totensor(img)
                label = self.totensor(label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            for img, label, border in zip(imgs, labels, borders):
                img = self.totensor(img)
                label = self.totensor(label)
                border = self.totensor(border)
                res_img.append(img)
                res_label.append(label)
                res_border.append(border)
            return res_img, res_label, res_border

class random_translate(object):
    def __init__(self, translate=[0.3, 0.3], p=0.5):
        '''
        For example translate=(a, b),
        then horizontal shift is randomly sampled in the range
        -img_width * a < dx < img_width * a
        and vertical shift is randomly sampled in the range
        -img_height * b < dy < img_height * b.
        '''
        self.translate = translate
        self.p = p

    def __call__(self, img, label, border=None):
        if random.random() < self.p:
            base_size = label.size

            max_dx = round(self.translate[0] * base_size[0])
            max_dy = round(self.translate[1] * base_size[1])
            tx = random.randint(-max_dx, max_dx)
            ty = random.randint(-max_dy, max_dy)
            translations = (tx, ty)

            # img = img.crop((-20, -20, 200, 200))
            h1 = translations[0]
            w1 = translations[1]
            h2 = h1 + base_size[0]
            w2 = w1 + base_size[1]

            img = img.crop((h1, w1, h2, w2))
            label = label.crop((h1, w1, h2, w2))
            if border is None:
                return img, label
            border = border.crop((h1, w1, h2, w2))
            return img, label, border

        if border is None:
            return img, label
        return img, label, border

class Random_vertical_flip(object):
    def _vertical_flip(self, img, label, border=None):
        # dsaFLIP_TOP_BOTTOM
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
        if border is None:
            return img, label
        border = border.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label, border

    def __init__(self, prob):
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label, border=None):
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._vertical_flip(img, label, border)
        else:
            if border is None:
                return img, label
            return img, label, border

class random_rotate(object):
    def __init__(self, range=[0, 360], interval=1, p=0.5):
        self.range = range
        self.interval = interval
        self.p = p

    def __call__(self, img, label, border=None):
        rot = (random.randint(*self.range) // self.interval) * self.interval
        rot = rot + 360 if rot < 0 else rot

        if random.random() < self.p:
            base_size = label.size
            img = img.rotate(rot, expand=True)
            label = label.rotate(rot, expand=True)

            img = img.crop(((img.size[0] - base_size[0]) // 2,
                            (img.size[1] - base_size[1]) // 2,
                            (img.size[0] + base_size[0]) // 2,
                            (img.size[1] + base_size[1]) // 2))

            label = label.crop(((label.size[0] - base_size[0]) // 2,
                                (label.size[1] - base_size[1]) // 2,
                                (label.size[0] + base_size[0]) // 2,
                                (label.size[1] + base_size[1]) // 2))
            if border is not None:
                border = border.rotate(rot, expand=True)
                border = border.crop(((border.size[0] - base_size[0]) // 2,
                                      (border.size[1] - base_size[1]) // 2,
                                      (border.size[0] + base_size[0]) // 2,
                                      (border.size[1] + base_size[1]) // 2))
        if border is None:
            return img, label
        return img, label, border

class random_enhance(object):
    def __init__(self, p=0.5, methods=['contrast', 'brightness', 'sharpness']):
        self.p = p
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, img, label, border=None):
        np.random.shuffle(self.enhance_method)
        for method in self.enhance_method:
            if np.random.random() < self.p:
                enhancer = method(img)
                factor = float(1 + np.random.random() / 10)
                img = enhancer.enhance(factor)

        if border is None:
            return img, label
        return img, label, border

class Random_horizontal_flip(object):
    def _horizontal_flip(self, img, label, border=None):
        # dsa
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if border is None:
            return img, label
        border = border.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label, border

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, img, label, border=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image), "should be PIL image"
        assert isinstance(label, Image.Image), "should be PIL image"
        if random.random() < self.prob:
            return self._horizontal_flip(img, label, border)
        else:
            if border is None:
                return img, label
            return img, label, border

