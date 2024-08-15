import copy
import torch
from torch import nn
from models.seg_net import Seg_Net
from models.losses import dice_bce_loss
from torch.nn import functional as F
from models.fformer_net import FFormer_Net


statistics = torch.load("/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth")     # "/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth"
mean = statistics['mean']
std = statistics['std']


def cal_dice(pred, gt):
    inter = (pred * gt).sum()
    dice = (2. * inter + 1) / (pred.sum() + gt.sum() + 1)
    return dice


class SegNet_FFormer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net_1 = Seg_Net()
        self.net_2 = FFormer_Net()
        self._get_net_2_encoder()
        
        # loss functions
        self.dice_bce_loss = dice_bce_loss
        
        # utils functions
        self.up_sample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.device = device
        self.total_step = 1
    
    def main_train(self, data_batch, seg_weght=1, temp_weight=2):
        """
        Args:
            Fs: [b,t,3,h,w], only the the 0-th frame is labeled & used as source image in flow computation
            Ms: [b,t,1,h,w]
        """
        Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
        b, T, c, h, w = Fs.shape
        
        # net_1
        seg_loss = torch.tensor(0).float().to(self.device)
        seg_pred_list, seg_cor_list = self.net_1(Fs)
        for pred in seg_pred_list:
            seg_loss += self.dice_bce_loss(pred[:, 0], Ms[:, 0])
        for pred in seg_cor_list:
            seg_loss += self.dice_bce_loss(pred[:, 0], Ms[:, 0])
            
        # net_2
        temp_loss = torch.tensor(0).float().to(self.device)
        temp_pred_list = self.net_2(Fs)
        
        for t in range(0, T - 1):
            gt_temporal = (temp_pred_list[0][:, t].sigmoid().detach() > 0.5).float()
            gt_seg =(seg_pred_list[0][:, t + 1].sigmoid().detach() > 0.5).float()
            
            # consistency loss
            for pred in temp_pred_list:
                temp_loss += self.dice_bce_loss(pred[:, t], gt_seg)
            for pred in seg_pred_list:
                temp_loss += self.dice_bce_loss(pred[:, t + 1], gt_temporal)
            for pred in seg_cor_list:
                temp_loss += self.dice_bce_loss(pred[:, t + 1], gt_temporal)
        temp_loss = temp_loss / (T - 1)
        
        loss = {
            "seg_loss": seg_loss,
            "temp_loss": temp_loss,
            "total_loss": seg_weght * seg_loss + temp_weight * temp_loss
        }
        pred = {
            "seg": seg_pred_list,
            "temp": temp_pred_list
        }
        
        return loss, pred
    
    def pre_train(self, data_batch, seg_weght=1, temp_weight=2):
        """
        Args:
            Fs: [b,t,3,h,w], only the the 0-th frame is labeled & used as source image in flow computation
            Ms: [b,t,1,h,w]
        """
        Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
        b, T, c, h, w = Fs.shape
        temp_loss = torch.tensor(0).float().to(self.device)
        seg_loss = torch.tensor(0).float().to(self.device)
        
        seg_pred_list, seg_cor_list = self.net_1(Fs)
        for t in range(T):
            for pred in seg_pred_list:
                seg_loss += self.dice_bce_loss(pred[:, t], Ms[:, t])
            for pred in seg_cor_list:
                seg_loss += self.dice_bce_loss(pred[:, t], Ms[:, t])

        temp_pred_list = self.net_2(Fs)
        for t in range(0, T - 1):
            for pred in temp_pred_list:
                temp_loss += self.dice_bce_loss(pred[:, t], Ms[:, t + 1])
        temp_loss = temp_loss / (T - 1)

        loss = {
            "seg_loss": seg_loss,
            "temp_loss": temp_loss,
            "total_loss": seg_weght * seg_loss + temp_weight * temp_loss
        }
        
        pred = {
            "seg": seg_pred_list,
            "temp": temp_pred_list
        }
        
        return loss, pred

    def _freeze_(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _train_(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
    def update_net_2_encoder(self):
        self._get_net_2_encoder()
    
    def _get_net_2_encoder(self):
        if self.net_2.encoder is None:
            self.net_2.encoder = copy.deepcopy(self.net_1.encoder)
        else:
            self._momentum_update_net(self.net_1.encoder, self.net_2.encoder)
        
        # freeze net_2 encoder
        for p in self.net_2.encoder.parameters():
            p.requires_grad = False
    
    def _momentum_update_net(self, net_q, net_k, m=0.99):
        """
        Momentum update of the key net
        """
        alpha = min(1 - 1 / (self.total_step + 1), m)
        if net_q is not None and net_k is not None:
            for param_q, param_k in zip(net_q.parameters(), net_k.parameters()):
                param_k.data = param_k.data * alpha + param_q.data * (1. - alpha)
        else:
            pass
        self.total_step += 1
    
    def evaluate(self, val_loader):
        cnt, dice = 0, 0
        with torch.no_grad():
            for data_batch in val_loader:
                Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
                pred_seg_list, _ = self.net_1(Fs)
                pred_seg = (pred_seg_list[0].sigmoid() > 0.5).float()
                b, t, c, h, w = pred_seg.shape
                for i in range(t):
                    pred = pred_seg[:, i, ...]
                    gt = Ms[:, i, ...]
                    dice += cal_dice(pred, gt)
                    cnt += 1
        out = {
            'dice': dice / cnt
        }
        return out


if __name__ == '__main__':
    Fs = torch.randn(8, 3, 3, 352, 352).to('cuda:1')
    Ms = torch.randn(8, 3, 1, 352, 352).to('cuda:1')
    m = SegNet_FFormer('cuda:1').to('cuda:1')