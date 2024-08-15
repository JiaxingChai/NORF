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


def eval(data_list, result_root):
    assert isinstance(data_list, list), "error"
    data_root = '/ssd/cjx/semi-supervised/data/SUN-SEG/SUN-SEG'
    result = {}
    for data in data_list:
        print(f'---------->eval on {data_mapper[data]}')
        dice, iou, sen, spe, cnt = 0, 0, 0, 0, 0
        gt_root = os.path.join(data_root, data_mapper[data], 'GT')
        pred_root = os.path.join(result_root, data_mapper[data])
        
        clips_list = os.listdir(f'{pred_root}/Pred')
        clips_list = [clip_name for clip_name in clips_list if 'Store' not in clip_name]
        for clip_name in tqdm(clips_list, ncols=130):
            clip_path = os.path.join(pred_root, 'Pred', clip_name)
            clip_img_names_list = os.listdir(clip_path)
            clip_img_names_list = [img_name for img_name in clip_img_names_list if img_name.endswith('.png')]
            
            for img_name in clip_img_names_list:
                gt_path =os.path.join(gt_root, clip_name, img_name)
                pred_path = os.path.join(pred_root, 'Pred', clip_name, img_name)
                
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
        
        result[data] = {'dice': dice / cnt,
                        'iou': iou / cnt,
                        'sen': sen / cnt,
                        'spe': spe / cnt,
                        }
        print(result[data])
    
    result['average'] = {'dice': sum([result[data]['dice'] for data in data_list]) / len(data_list),
                         'iou': sum([result[data]['iou'] for data in data_list]) / len(data_list),
                         'sen': sum([result[data]['sen'] for data in data_list]) / len(data_list),
                         'spe': sum([result[data]['spe'] for data in data_list]) / len(data_list),
                        }
                
    return result


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='mySemi')
    args.add_argument('--model-name', type=str, default='setting_5')
    args.add_argument('--t', type=str)
    args = args.parse_args()
    
    data_list = ['test_easy_seen', 'test_easy_unseen', 'test_hard_seen', 'test_hard_unseen']
    # data_list = ['train']
    result_root = f'outputs/{args.model_name}/{args.t}/predictions/'
    print(f'\n ------>> eval on {result_root} \n')
    ret = eval(data_list, result_root)
    print(ret)