import torch
import os
from torch.utils.data import DataLoader
from datasets.func import get_test_datasetV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.models.setting_5 import SegNet_FFormer as Setting_5
from PIL import Image
import numpy as np


import warnings
warnings.filterwarnings("ignore")

"""
python MyTest.py --load your_model
"""

data_mapper = {
    "test_easy_seen": "TestEasyDataset/Seen/",
    "test_easy_unseen": "TestEasyDataset/Unseen/",
    "test_hard_seen": "TestHardDataset/Seen/",
    "test_hard_unseen": "TestHardDataset/Unseen/",
}

def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img


class AutoTest:
    def __init__(self, args, test_dataset):
        assert isinstance(test_dataset, list), "error"
        self.test_dataset = test_dataset
        self.dataloader = {}
        for dst in self.test_dataset:
            loader = get_test_datasetV2(img_size=352, test_root=dst, frames_per_clip=10, memory_size=3, type='normal')
            self.dataloader[dst] = DataLoader(loader, batch_size=1, shuffle=False, num_workers=8)
            
        self.model = Setting_5(args.device).to(args.device)
        self.model.load_state_dict(torch.load(f'outputs/{args.model_name}/{args.t}/main_train/{args.load}'))
        self.tag_dir = f'outputs/{args.model_name}/{args.t}/predictions/'
        self.data_root = '/ssd/cjx/semi-supervised/data/SUN-SEG/SUN-SEG'
        self.model.eval()

    def test(self):
        print('Saving to ', self.tag_dir)
        with torch.no_grad():
            for dst in self.test_dataset:
                for data in tqdm(self.dataloader[dst], desc="test:%s" % dst):
                    img_li = data['img'].to(args.device)
                    path_li = data['path']

                    result_list, _ = self.model.net_1(img_li)
                    result = torch.sigmoid(result_list[0])
                    b, t, c, h, w = result.shape

                    for i in range(t):
                        result_i = result[:, i, ::]
                        path_i = path_li[i]

                        for idx in range(b):
                            res_idx = result_i[idx]
                            path = path_i[idx]
                            npres = res_idx.squeeze().cpu().numpy()
                            
                            save_path = path.replace(self.data_root, self.tag_dir).replace('Frame', 'Pred').replace('.jpg', '.png')
                            safe_save(Image.fromarray((npres * 255).astype(np.uint8)), save_path)


if __name__ == "__main__":
    test_list = ["test_easy_seen", "test_easy_unseen", "test_hard_seen", "test_hard_unseen"]
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="best_eval_model.pth")
    parser.add_argument("--model-name", type=str, default="setting_5")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument('--t', type=str)
    args = parser.parse_args()
    
    at = AutoTest(args, test_list)
    at.test()
