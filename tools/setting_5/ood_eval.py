import numpy
import os
from tqdm import tqdm
from PIL import Image

img_size = 352
data_mapper = {
    "test_easy_seen": "TestEasyDataset/Seen/",
    "test_easy_unseen": "TestEasyDataset/Unseen/",
    "test_hard_seen": "TestHardDataset/Seen/",
    "test_hard_unseen": "TestHardDataset/Unseen/",
    "train": 'TrainDataset/'
}

def calu_dice(pred, gt):
    return 2 * numpy.sum(pred * gt) / (numpy.sum(pred) + numpy.sum(gt))

def calu_iou(pred, gt):
    return numpy.sum(pred * gt) / numpy.sum(pred + gt - pred * gt)

def calu_sen(pred, gt): # sensitivity
    return numpy.sum(pred * gt) / numpy.sum(gt)

def calu_spe(pred, gt): # specificity
    return numpy.sum((1 - pred) * (1 - gt)) / numpy.sum(1 - gt)

def eval(gt_root, result_root):
    images_list = os.listdir(gt_root)
    dice, iou, sen, spe, cnt = 0, 0, 0, 0, 0
    for image_name in tqdm(images_list, ncols=130):
        gt_path = os.path.join(gt_root, image_name)
        pred_path = os.path.join(result_root, image_name)

        gt = numpy.array(Image.open(gt_path).convert('L').resize((img_size, img_size)))
        pred = numpy.array(Image.open(pred_path).convert('L').resize((img_size, img_size)))

        gt = gt / gt.max() if gt.max() != 0 else gt
        pred = pred / pred.max() if pred.max() != 0 else pred

        if pred.max() == 0:
            dice += 0
            iou += 0
            sen += 0
            spe += 0
        else:
            dice += calu_dice(pred, gt)
            iou += calu_iou(pred, gt)
            sen += calu_sen(pred, gt)
            spe += calu_spe(pred, gt)
        cnt += 1
    
    result = {'dice': dice / cnt,
              'iou': iou / cnt,
              'sen': sen / cnt,
              'spe': spe / cnt,
                        }
    return result

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='mySemi')
    args.add_argument('--model-name', type=str, default='setting_5')
    args.add_argument('--t', type=str)
    args = args.parse_args()
    datasets = ['CVC-ClinicDB', 'Kvasir']
    for dataset in datasets:
        result_root = f'/ssd/cjx/semi-supervised/tcnet/outputs/OOD/{args.model_name}/{args.t}/{dataset}/'
        print(f'\n ------>> eval on {result_root} \n')
        gt_root = f'/ssd/cjx/semi-supervised/data/{dataset}/TestDataset/masks/'
        ret = eval(gt_root, result_root)
        print(ret)