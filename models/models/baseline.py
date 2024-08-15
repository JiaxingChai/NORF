import copy
import torch
from torch import nn
from models.seg_net import Base_Net
from models.losses import dice_bce_loss
from torch.nn import functional as F


statistics = torch.load("/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth")     # "/ssd/cjx/semi-supervised/VPS/lib/dataloader/statistics.pth"
mean = statistics['mean']
std = statistics['std']

def cal_dice(x, y):
    inter = (x * y).sum()
    dice = (2. * inter + 1) / (x.sum() + y.sum() + 1)
    return dice

class BaseNet_BaseNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        """
        branch 1: base_net
        branch 2: base_net (encoder from branch_1, optimized through MAE)
        """
        self.net_1 = Base_Net()
        self.net_2 = None
        self._get_temporal_encoder()
        
        # loss functions
        self.dice_bce_loss = dice_bce_loss
        
        # utils functions
        self.up_sample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.device = device
        self.total_step = 1
    
    def main_train(self, data_batch):
        """
        Args:
            Fs: [b,t,3,h,w], only the the 0-th frame is labeled
            Ms: [b,t,1,h,w]
        """
        Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
        b, T, c, h, w = Fs.shape
        
        # seg net
        seg_pred_list = self.net_1(Fs)
        # temporal net
        temp_pred_list = self.net_2(Fs)
        
        # supervised loss
        seg_loss = torch.tensor(0).float().to(self.device)
        for pred in seg_pred_list:
            seg_loss += self.dice_bce_loss(pred[:, 0], Ms[:, 0])
        
        # consistency loss
        temp_loss = torch.tensor(0).float().to(self.device)
        for t in range(1, T):
            gt_temporal = (temp_pred_list[0][:, t].detach().sigmoid() > 0.5).float()
            gt_seg = (seg_pred_list[0][:, t].detach().sigmoid() > 0.5).float()
            
            for pred in temp_pred_list:
                temp_loss += self.dice_bce_loss(pred[:, t], gt_seg)
            for pred in seg_pred_list:
                temp_loss += self.dice_bce_loss(pred[:, t], gt_temporal)
        temp_loss = temp_loss / (T - 1)
        
        loss = {
            "seg_loss": seg_loss,
            "temp_loss": temp_loss,
            "total_loss": seg_loss + temp_loss
        }
        pred = {
            "seg": seg_pred_list,
            "temp": temp_pred_list
        }
        
        return loss, pred
    
    def pre_train(self, data_batch):
        """
        Args:
            Fs: [b,t,3,h,w], only the the 0-th frame is labeled
            Ms: [b,t,1,h,w]
        """
        # we only train the seg_net in the pre-training stage
        self._freeze_(self.net_2)
        self._train_(self.net_1)
        
        Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
        b, T, c, h, w = Fs.shape
        
        # seg net
        seg_loss = torch.tensor(0).float().to(self.device)
        seg_pred_list = self.net_1(Fs)
        for t in range(T):
            for pred in seg_pred_list:
                seg_loss += self.dice_bce_loss(pred[:, t], Ms[:, t])
            
        loss = {
            "seg_loss": seg_loss
        }
        
        pred = {
            "seg": seg_pred_list
        }
        
        return loss, pred
    
    def evaluate(self, val_loader):
        cnt, dice = 0, 0
        with torch.no_grad():
            for data_batch in val_loader:
                Fs, Ms = data_batch['img'].to(self.device), data_batch['mask'].to(self.device)
                pred_seg_list = self.net_1(Fs)
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
        
    def _freeze_(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def _train_(self, model):
        for param in model.parameters():
            param.requires_grad = True
            
    def update_temporal_encoder(self):
        self._get_temporal_encoder()
    
    def _get_temporal_encoder(self):
        if self.net_2 is None:
            self.net_2 = copy.deepcopy(self.net_1)
        elif self.net_2.encoder is None:
            self.net_2.encoder = copy.deepcopy(self.net_1.encoder)
        else:
            self._momentum_update_net(self.net_1.encoder, self.net_2.encoder)
        
        # freeze temporal net encoder
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

if __name__ == '__main__':
    Fs = torch.randn(8, 3, 3, 352, 352).cuda()
    Ms = torch.randn(8, 3, 1, 352, 352).cuda()
    m = BaseNet_BaseNet('cuda').cuda()
