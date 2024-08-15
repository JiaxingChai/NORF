from cgi import test
import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from models.models.setting_5 import SegNet_FFormer as Setting_5
from PIL import Image
import numpy as np


import warnings
warnings.filterwarnings("ignore")

"""
python MyTest.py --load your_model
"""

def safe_save(img, save_path):
    os.makedirs(save_path.replace(save_path.split('/')[-1], ""), exist_ok=True)
    img.save(save_path)

class PolypDataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, testsize=352):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class AutoTest:
    def __init__(self, args, test_dataset):
        assert isinstance(test_dataset, str), "error"
        self.test_dataset = f'/ssd/cjx/semi-supervised/data/{test_dataset}/TestDataset/images/'
        self.loader = PolypDataset(self.test_dataset)
        self.tag_dir = f'outputs/OOD/{args.model_name}/{args.t}/{test_dataset}/'
        if not os.path.exists(self.tag_dir):
            os.makedirs(self.tag_dir)
            
        model = Setting_5(args.device).to(args.device)
        model.load_state_dict(torch.load(f'outputs/{args.model_name}/{args.t}/main_train/{args.load}'))
        self.seg_net = model.net_1
        self.seg_net.eval()
        del model

    def test(self):
        print('Saving to ', self.tag_dir)
        with torch.no_grad():
            for i in range(self.loader.size):
                image, name = self.loader.load_data()
                image = image.unsqueeze(0).repeat(1, 3, 1, 1, 1)
                image = image.to(args.device)
                result_list, _ = self.seg_net(image)
                result = torch.sigmoid(result_list[0])[:, 0]
                result = result.squeeze().cpu().numpy()
                result = (result * 255).astype(np.uint8)
                result = Image.fromarray(result)
                result.save(os.path.join(self.tag_dir, name))

if __name__ == "__main__":
    import argparse
    test_datasets =['CVC-ClinicDB', 'Kvasir']

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="best_eval_model.pth")
    parser.add_argument("--model-name", type=str, default="setting_5")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument('--t', type=str)
    args = parser.parse_args()
    for test_dataset in test_datasets:
        at = AutoTest(args, test_dataset)
        at.test()
