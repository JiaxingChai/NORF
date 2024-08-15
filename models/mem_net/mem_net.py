import torch
from torch import nn
from torch.nn import functional as F
from models.mem_net.memory import get_affinity, readout
from models.mem_net.modules import QKV_Encoder
from models.mem_net.cbam import CBAM


def CBR(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, 3, 1, 1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True)
    )


class base_Decoder(nn.Module):
    def __init__(self, in_chs) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            CBR(2 * in_chs, in_chs),
            CBR(in_chs, in_chs),
            CBR(in_chs, in_chs),
            CBR(in_chs, in_chs),
            nn.Conv2d(in_chs, 1, 1)
        )
        self.interp = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    
    def forward(self, feat, mask):
        mask = self.interp(mask, feat.shape[-2:])
        re_feat = 1 - torch.sigmoid(feat)
        feat = torch.cat([feat, re_feat], dim=1)
        out_mask = self.conv(feat)
        
        return out_mask


class Decoder(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.mem_fuse = nn.Sequential(
            CBR(64, 32),
            CBAM(32, reduction_ratio=4),
            CBR(32, 32),
        )
        self.decoder_32 = base_Decoder(32)
        self.decoder_16 = base_Decoder(32)
        self.decoder_8 = base_Decoder(32)
    
    def forward(self, f8_t, f16_t, f32_t, m_t, mem_feat):
        f32_t = torch.cat([f32_t, mem_feat], dim=1)
        f32_t = self.mem_fuse(f32_t)
        
        m32 = self.decoder_32(f32_t, m_t)
        m16 = self.decoder_16(f16_t, m32)
        m8 = self.decoder_8(f8_t, m16)
        
        return m8, m16, m32


class Mem_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.qkv_proj = QKV_Encoder(32, 32)
        self.interp = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        self.decoder = Decoder()
        self.up_sample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    
    def forward(self, Fs, Ms):
        b, T, c, h, w = Fs.shape
        with torch.no_grad():
            out8, a, [f8_32, f16_32, f32_32], [f8, f16, f32] = self.encoder(Fs.reshape(b * T, c, h, w))
            out8 = out8.reshape(b, T, *out8.shape[1:]).detach()
            f8_32 = f8_32.reshape(b, T, *f8_32.shape[1:]).detach()
            f16_32 = f16_32.reshape(b, T, *f16_32.shape[1:]).detach()
            f32_32 = f32_32.reshape(b, T, *f32_32.shape[1:]).detach()
        
        proj, key_scale, query_selects, value = self.qkv_proj(f32_32, Fs[:, 0], Ms[:, 0])    # b t c h w, b t 1 h w, b t c h w, b c h w
        
        # reference key and key_scale
        key = proj[:, 0]
        key_scale = key_scale[:, 0]
        
        m2_out, m3_out, m4_out = [], [], []
        
        for t in range(1, T):
            f8_t, f16_t, f32_t, out8_t = f8_32[:, t], f16_32[:, t], f32_32[:, t], out8[:, t]
            query = proj[:, t]
            query_select = query_selects[:, t]
            affinity = get_affinity(key, key_scale, query, query_select)
            mem_feat = readout(affinity, value)       # 2 32 11 11
            m2_t, m3_t, m4_t = self.decoder(f8_t, f16_t, f32_t, Ms[:, t], mem_feat)
            m2_out.append(self.up_sample(m2_t, size=(h, w)))
            m3_out.append(self.up_sample(m3_t, size=(h, w)))
            m4_out.append(self.up_sample(m4_t, size=(h, w)))
        
        m2_out = torch.stack(m2_out, dim=1)
        m3_out = torch.stack(m3_out, dim=1)
        m4_out = torch.stack(m4_out, dim=1)
        
        return m2_out, m3_out, m4_out


if __name__ == '__main__':
    import copy
    from models.seg_net.seg_net import Seg_Net
    seg_net = Seg_Net()
    mem_net = Mem_Net()
    mem_net.encoder = copy.deepcopy(seg_net.encoder)
    
    x = torch.randn(2, 4, 3, 352, 352)
    mask = torch.randn(2, 4, 1, 352, 352)
    m2_out, m3_out, m4_out = mem_net(x, mask)
    print(m2_out.shape, m3_out.shape, m4_out.shape)