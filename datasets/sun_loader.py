""" 
we provide the dataloader of SUN-SEG dataset for semi-supervised segmentation
modified on the basis of https://github.com/GewelsJI/VPS/blob/main/lib/dataloader/dataloader.py
"""
import os
import cv2
import torch
import warnings
import random
import json
from PIL import Image
from torch.utils.data import Dataset
from datasets.preprocess import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

data_root = "/ssd/cjx/semi-supervised/data/SUN-SEG/SUN-SEG"

test_mapper = {
    'test_easy_seen': 'TestEasyDataset/Seen/',
    'test_easy_unseen': 'TestEasyDataset/Unseen/',
    'test_hard_seen': 'TestHardDataset/Seen/',
    'test_hard_unseen': 'TestHardDataset/Unseen/',
}

class TrainDataset(Dataset):
    def __init__(self, transform, video_time_clips=15, type='normal'):
        super(TrainDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []

        video_root = os.path.join(data_root, "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
    
        cls_list = json.load(json_file)['train']
        print(f'------>> {type} train cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            idx_matrix = self.create_idx_metrix(len(li), video_time_clips)
            for idx in idx_matrix:
                clip = [li[int(i.item())] for i in idx]
                self.video_train_list.append(clip)
            
        self.img_label_transform = transform
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        # img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border=border_li)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, _) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                # BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            # "border": BORDER
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class TrainDatasetV2(Dataset):
    def __init__(self, transform, video_time_clips=10, memory_size=3, type='normal'):
        super(TrainDatasetV2, self).__init__()
        self.time_clips = video_time_clips
        self.memory_size = memory_size
        self.video_train_list = []

        video_root = os.path.join(data_root, "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)['train']
        print(f'------>> {type} train cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            for begin in range(0, len(li), self.time_clips):
                for i in range((len(li) - begin)//(self.memory_size + 1) + 1):
                    # reference frame
                    clip = [li[begin]]
                    
                    # other reference frames
                    # assert no other reference frames are selected
                    # selected_list = range(begin+1, len(li))
                    selected_list = list(set(range(begin + 1, len(li))) - set(range(0, len(li), self.time_clips)))
                    if len(selected_list) >= self.memory_size - 1:
                        select = random.sample(selected_list, self.memory_size - 1)
                    else:
                        selected_list = list(set(range(0, len(li))) - set(range(0, len(li), self.time_clips)))
                        select = random.sample(selected_list, self.memory_size - 1)
                    select.sort()
                    
                    for id in select:
                        clip.append(li[id])
                    self.video_train_list.append(clip)
            
        self.img_label_transform = transform

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border_li)
        # img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, border) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            "border": BORDER,
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class TestDataset(Dataset):
    def __init__(self, test_root, transform, video_time_clips=10, type='normal'):
        super(TestDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []

        video_root = os.path.join(data_root, test_mapper[test_root])
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)[test_root]
        print(f'------>> {type} {test_root} cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out non-jpg files
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            idx_matrix = self.create_idx_metrix(len(li), video_time_clips)
            for idx in idx_matrix:
                clip = [li[int(i.item())] for i in idx]
                self.video_train_list.append(clip)
            
        self.img_label_transform = transform
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        # img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border=border_li)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, _) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                # BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            # "border": BORDER
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class TestDatasetV2(Dataset):
    def __init__(self, test_root, transform, video_time_clips=10, memory_size=3, type='normal'):
        super(TestDatasetV2, self).__init__()
        self.time_clips = video_time_clips
        self.memory_size = memory_size
        self.video_train_list = []

        video_root = os.path.join(data_root, test_mapper[test_root])
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)[test_root]
        print(f'------>> {type} {test_root} cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out non-jpg files
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            for begin in range(0, len(li), self.memory_size):
                clip = []
                for t in range(self.memory_size):
                    if begin + t >= len(li):
                        clip.insert(0, li[begin - 1 - (begin + t - len(li))])
                    else:
                        clip.append(li[begin + t])
                self.video_train_list.append(clip)
                    
        self.img_label_transform = transform
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        # img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border=border_li)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, _) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                # BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            # "border": BORDER
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class ValDataset(Dataset):
    def __init__(self, transform, video_time_clips=10, type='normal'):
        super(ValDataset, self).__init__()
        self.time_clips = video_time_clips
        self.video_train_list = []

        video_root = os.path.join(data_root, 'TrainDataset')
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)['val']
        print(f'------>> {type} val cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out non-jpg files
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            idx_matrix = self.create_idx_metrix(len(li), video_time_clips)
            for idx in idx_matrix:
                clip = [li[int(i.item())] for i in idx]
                self.video_train_list.append(clip)
                    
        self.img_label_transform = transform
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        # img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border=border_li)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, _) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                # BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            # "border": BORDER
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class ValDatasetV2(Dataset):
    def __init__(self, transform, video_time_clips=10, memory_size=3, type='normal'):
        super(ValDatasetV2, self).__init__()
        self.time_clips = video_time_clips
        self.memory_size = memory_size
        self.video_train_list = []

        video_root = os.path.join(data_root, 'TrainDataset')
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)['val']
        print(f'------>> {type} val cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out non-jpg files
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            for begin in range(0, len(li), self.memory_size):
                clip = []
                for t in range(self.memory_size):
                    if begin + t >= len(li):
                        clip.insert(0, li[begin - 1 - (begin + t - len(li))])
                    else:
                        clip.append(li[begin + t])
                self.video_train_list.append(clip)
                    
        self.img_label_transform = transform
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix

    def __getitem__(self, idx):
        img_label_li = self.video_train_list[idx]
        IMG = None
        LABEL = None
        img_li = []
        label_li = []
        border_li = []
        for idx, (img_path, label_path, border_path) in enumerate(img_label_li):
            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            border = Image.open(border_path).convert('L')
            img_li.append(img)
            label_li.append(label)
            border_li.append(border)
        # img_li, label_li, border_li = self.img_label_transform(img_li, label_li, border=border_li)
        img_li, label_li = self.img_label_transform(img_li, label_li)
        for idx, (img, label, _) in enumerate(zip(img_li, label_li, border_li)):
            if idx == 0:
                IMG = torch.zeros(len(img_li), *(img.shape))
                LABEL = torch.zeros(len(img_li), *(label.shape))
                # BORDER = torch.zeros(len(img_li), *(label.shape))

                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
            else:
                IMG[idx, :, :, :] = img
                LABEL[idx, :, :, :] = label
                # BORDER[idx, :, :, :] = border
        
        data = {
            "img": IMG,
            "mask": LABEL,
            # "border": BORDER
            "path": [img_path for img_path, _, _ in img_label_li]
        }

        return data

    def __len__(self):
        return len(self.video_train_list)


class SUN_PesudoDataset(Dataset):
    def __init__(self, img_size, video_time_clips=15, type='normal'):
        super().__init__()
        self.time_clips = video_time_clips
        self.img_size = img_size
        self.video_train_list = []

        video_root = os.path.join(data_root, "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)['train']
        print(f'------>> {type} train Pseudo cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            idx_matrix = self.create_idx_metrix(len(li), video_time_clips)
            for idx in idx_matrix:
                self.video_train_list.append(li[int(idx[0].item())])
                
        statistics = torch.load("/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth")
        self.transform = Compose_imglabel([toTensor_video(),
                                           Normalize_video(statistics["mean"], statistics["std"])
                                           ])
        self.affine_transformer = []
        for i in range(self.time_clips - 1):
            translate = random.uniform(0, 0.5)
            trans = Compose_imglabel([
                random_translate(translate=[translate, translate], p=0.7),
                Random_horizontal_flip(0.5),
                Random_vertical_flip(0.5),
                random_rotate(range=[0, 359], p=0.7),
                random_enhance(p=0.5)
            ])
            self.affine_transformer.append(trans)
    
    def create_idx_metrix(self, length, num_clips):
        num_segments = length // num_clips
        remainder = length % num_clips

        idx_matrix = torch.zeros((num_segments+1, num_clips))
        for i in range(num_clips):
            if i < remainder:
                idx = torch.arange(i*(num_segments+1), (i+1)*(num_segments+1), dtype=torch.int64)
                idx_matrix[:, i] = idx
            else:
                idx = torch.arange(i*num_segments+remainder, (i+1)*num_segments+remainder, dtype=torch.int64)
                idx_matrix[:-1, i] = idx
                idx_matrix[-1, i] = random.choice(idx)
        
        return idx_matrix
    
    def __getitem__(self, index):
        img_li, label_li = [], []
        frame = self.video_train_list[index]
        img_path, label_path, border_path = frame

        img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        label = Image.open(label_path).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
        
        img_li.append(img)
        label_li.append(label)
        
        for i in range(self.memory_size - 1):
            aff_img, aff_label = self.affine_transformer[i](img, label)
            img_li.append(aff_img)
            label_li.append(aff_label)
        
        img_li, label_li = self.transform(img_li, label_li)
        
        data = {
            'img': torch.stack(img_li, dim=0),
            'mask': torch.stack(label_li, dim=0),
            'path': img_path,
        }
        
        return data
    
    def __len__(self):
        return len(self.video_train_list)


class SUN_PesudoDatasetV2(Dataset):
    def __init__(self, img_size, video_time_clips=15, memory_size=3, type='normal'):
        super().__init__()
        self.time_clips = video_time_clips
        self.memory_size = memory_size
        self.img_size = img_size
        self.video_train_list = []

        video_root = os.path.join(data_root, "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        cls_list = json.load(json_file)['train']
        print(f'------>> {type} train Pseudo cases: ', cls_list)
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            for begin in range(0, len(li), self.time_clips):
                self.video_train_list.append(li[begin])
                
        statistics = torch.load("/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth")
        self.transform = Compose_imglabel([toTensor_video(),
                                           Normalize_video(statistics["mean"], statistics["std"])
                                           ])
        self.affine_transformer = []
        for i in range(self.memory_size - 1):
            translate = random.uniform(0, 0.5)
            trans = Compose_imglabel([
                random_translate(translate=[translate, translate], p=0.7),
                Random_horizontal_flip(0.5),
                Random_vertical_flip(0.5),
                random_rotate(range=[0, 359], p=0.7),
                random_enhance(p=0.5)
            ])
            self.affine_transformer.append(trans)
    
    def __getitem__(self, index):
        img_li, label_li = [], []
        frame = self.video_train_list[index]
        img_path, label_path, border_path = frame

        img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        label = Image.open(label_path).convert('L').resize((self.img_size, self.img_size), Image.NEAREST)
        
        img_li.append(img)
        label_li.append(label)
        
        for i in range(self.memory_size - 1):
            aff_img, aff_label = self.affine_transformer[i](img, label)
            img_li.append(aff_img)
            label_li.append(aff_label)
        
        img_li, label_li = self.transform(img_li, label_li)
        
        data = {
            'img': torch.stack(img_li, dim=0),
            'mask': torch.stack(label_li, dim=0),
            'path': img_path,
        }
        
        return data
    
    def __len__(self):
        return len(self.video_train_list)


class TrainDatasetV2_image(Dataset):
    def __init__(self, video_time_clips, mode='train', type='normal', sup=True, img_size=352) -> None:
        super().__init__()
        assert mode in ['train', 'val']
        self.mode = mode
        self.sup = sup
        self.time_clips = video_time_clips
        self.video_train_list = []

        video_root = os.path.join(data_root, "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join(data_root, f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        self.train_sup_li = []
        self.train_unsup_li = []
        self.val_li = []
        
        # get train_sup_dataset, train_unsup_dataset
        cls_list = json.load(json_file)[mode]
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            for idx in range(0, len(li)):
                if mode == 'train':
                    if idx // self.time_clips == 0:
                        self.train_sup_li.append(li[idx])
                    else:
                        self.train_unsup_li.append(li[idx])
                elif mode == 'val':
                    self.val_li.append(li[idx])
        
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                ToTensorV2()
            ])
        print(f'------>> {mode} on {type} cases: ', cls_list)
        print(f'train supervised images: {len(self.train_sup_li)}, train unsupervised images: {len(self.train_unsup_li)}') \
            if mode == 'train' else print(f'validate images: {len(self.val_li)}')
    
    def __getitem__(self, index):
        if self.mode == 'train' and self.sup is True:
            data = self.train_sup_li[index]
        if self.mode == 'train' and self.sup is False:
            data = self.train_unsup_li[index]
        if self.mode == 'val':
            data = self.val_li[index]
        
        image_path, label_path, border_path = data
        image = self.rgb_loader(image_path)
        mask = self.binary_loader(label_path)
        
        sample = self.transform(image=image, mask=mask)
        
        return sample['image'], sample['mask']/sample['mask'].max()
    
    def rgb_loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def binary_loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    
    def __len__(self):
        if self.mode == 'train' and self.sup is True:
            return len(self.train_sup_li)
        if self.mode == 'train' and self.sup is False:
            return len(self.train_unsup_li)
        if self.mode == 'val':
            return len(self.val_li)


# designed for image-based semi-supervised methods
class TrainDatasetV2_image(Dataset):
    def __init__(self, video_time_clips, mode='train', type='normal', sup=True, img_size=352) -> None:
        super().__init__()
        assert mode in ['train', 'val']
        self.mode = mode
        self.sup = sup
        self.time_clips = video_time_clips
        self.video_train_list = []
        
        self.num_classes = 1
        self.ignore_index = 255
        self.palette = [0, 0, 0, 255, 255, 255]
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        video_root = os.path.join("/ssd/cjx/semi-supervised/data/SUN-SEG/SUN-SEG", "TrainDataset")
        img_root = os.path.join(video_root, 'Frame')
        gt_root = os.path.join(video_root, 'GT')
        border_root = os.path.join(video_root, 'Edge')

        json_root = os.path.join("/ssd/cjx/semi-supervised/data/SUN-SEG/SUN-SEG", f'SUN-SEG_{type}.json')
        json_file = open(json_root, 'r')
        
        self.train_sup_li = []
        self.train_unsup_li = []
        self.val_li = []
        
        # get train_sup_dataset, train_unsup_dataset, val_dataset
        cls_list = json.load(json_file)[mode]
        self.video_filelist = {}
        for cls in cls_list:
            self.video_filelist[cls] = []

            cls_img_path = os.path.join(img_root, cls)
            cls_label_path = os.path.join(gt_root, cls)
            cls_border_path = os.path.join(border_root, cls)

            tmp_list = os.listdir(cls_img_path)
            tmp_list = [i for i in tmp_list if 'Store' not in i]  # filter out .DS_Store
            tmp_list.sort(key=lambda name: (
                int(name.split('-')[0].split('_')[-1]),
                int(name.split('_a')[1].split('_')[0]),
                int(name.split('_image')[1].split('.jpg')[0])))

            for filename in tmp_list:
                self.video_filelist[cls].append((
                    os.path.join(cls_img_path, filename),
                    os.path.join(cls_label_path, filename.replace(".jpg", ".png")),
                    os.path.join(cls_border_path, filename.replace(".jpg", ".png")),
                ))
        # ensemble
        
        for cls in cls_list: # case1_1, case1_3, ...
            li = self.video_filelist[cls]   # all frames in a video clip
            print('---> len clips li:', len(li))
            for idx in range(0, len(li)):
                if mode == 'train':
                    if idx % self.time_clips == 0:     # supervised frame
                        self.train_sup_li.append(li[idx])
                        print('sup:', idx)
                    else:
                        self.train_unsup_li.append(li[idx])     # un-supervised frame
                elif mode == 'val':
                    self.val_li.append(li[idx])
        
        if self.mode == 'train':
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                ToTensorV2()
            ])
        print(f'------>> {mode} on {type} cases: ', cls_list)
        print(f'train supervised images: {len(self.train_sup_li)}, train unsupervised images: {len(self.train_unsup_li)}') \
            if mode == 'train' else print(f'validate images: {len(self.val_li)}')
    
    def __getitem__(self, index):
        if self.mode == 'train' and self.sup is True:
            data = self.train_sup_li[index]
        if self.mode == 'train' and self.sup is False:
            data = self.train_unsup_li[index]
        if self.mode == 'val':
            data = self.val_li[index]
        
        image_path, label_path, border_path = data
        image = self.rgb_loader(image_path)
        mask = self.binary_loader(label_path)
        
        sample = self.transform(image=image, mask=mask)
        
        return sample['image'], sample['mask'] / sample['mask'].max()
    
    def rgb_loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def binary_loader(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    
    def __len__(self):
        if self.mode == 'train' and self.sup is True:
            return len(self.train_sup_li)
        if self.mode == 'train' and self.sup is False:
            return len(self.train_unsup_li)
        if self.mode == 'val':
            return len(self.val_li)