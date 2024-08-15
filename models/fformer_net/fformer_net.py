from detectron2.utils import memory
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.seg_net.encoders.res2net_encoder import res2net_Encoder
from models.fformer_net.transformer import CrossAttentionLayer, SelfAttentionLayer, FFNLayer, PositionEmbeddingSine


def base_conv(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_chs)
    )


class base_Decoder(nn.Module):
    def __init__(self, d_model, nheads):
        super().__init__()
        self.nheads = nheads
        self.pe_layer = PositionEmbeddingSine(num_pos_feats=d_model//2)
        self.cross_attn = CrossAttentionLayer(d_model=d_model, nhead=nheads)
        self.self_attn = SelfAttentionLayer(d_model=d_model, nhead=nheads)
        self.ffn = FFNLayer(d_model=d_model, dim_feedforward=d_model*4)
        self.up_sample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        self.conv_out = nn.Sequential(
            # base_conv(2 * d_model, d_model),
            base_conv(3 * d_model, d_model),
            nn.ReLU(),
            base_conv(d_model, d_model),
            nn.ReLU(),
            base_conv(d_model, d_model),
            nn.ReLU(),
            base_conv(d_model, 1)
        )

    def forward(self, F_t: torch.Tensor, F_r: torch.Tensor, m_t: torch.Tensor):
        """_summary_

        Args:
            F_t (torch.Tensor): current frame feature
            F_r (torch.Tensor): reference frame feature
            m_t (torch.Tensor): mask for current frame from deeper decoder block
            m_r (torch.Tensor): mask for reference frame
        """
        b, c, h, w = F_t.shape
        
        t_pos = self.pe_layer(F_t).flatten(2).permute(0, 2, 1).contiguous()
        p_pos = self.pe_layer(F_r).flatten(2).permute(0, 2, 1).contiguous()
        t_flatten = F_t.flatten(2).permute(0, 2, 1).contiguous()
        p_flatten = F_r.flatten(2).permute(0, 2, 1).contiguous()
        # cross attention
        out = self.cross_attn(tgt=p_flatten, memory=t_flatten, query_pos=p_pos, pos=t_pos)
        # out = self.cross_attn(tgt=t_flatten, memory=p_flatten, query_pos=t_pos, pos=p_pos)

        # self attention
        out = self.self_attn(out, query_pos=t_pos)
        # ffn layer
        out = self.ffn(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        rev_out = out * (1 - self.up_sample(m_t, size=(h, w)))
        out = torch.cat([F_t, rev_out, out], dim=1)
        out = self.conv_out(out)
        
        return out


class Former_Decoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.decoder_2 = base_Decoder(d_model=32, nheads=8,)
        self.decoder_3 = base_Decoder(d_model=32, nheads=8,)
        self.decoder_4 = base_Decoder(d_model=32, nheads=8,)
    
    def forward(self, x2_t, x3_t, x4_t, m5_t, x2_r, x3_r, x4_r):
        m4_t = self.decoder_4(x4_t, x4_r, m5_t,)
        m3_t = self.decoder_3(x3_t, x3_r, m4_t,)
        m2_t = self.decoder_2(x2_t, x2_r, m3_t,)

        return m2_t, m3_t, m4_t


class FFormer_Net(nn.Module):
    """
    branch 2
    frame_former decoder
    """
    def __init__(self,):
        super().__init__()
        self.encoder = None # encoder is got from net_1 by deepcopy
        self.decoder = Former_Decoder()
        self.up_sample = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
    
    def forward(self, Fs):
        b, T, c, h, w = Fs.shape
        with torch.no_grad():
            out, a, [x2_cnt, x3_cnt, x4_cnt], [x2, x3, x4] = self.encoder(Fs.reshape(b * T, c, h, w))
            out = out.reshape(b, T, *out.shape[1:]).detach()
            x2_cnt = x2_cnt.reshape(b, T, *x2_cnt.shape[1:]).detach()
            x3_cnt = x3_cnt.reshape(b, T, *x3_cnt.shape[1:]).detach()
            x4_cnt = x4_cnt.reshape(b, T, *x4_cnt.shape[1:]).detach()
        
        m2_out, m3_out, m4_out = [], [], []
        x2_r, x3_r, x4_r = x2_cnt[:, 0], x3_cnt[:, 0], x4_cnt[:, 0]
        for t in range(1, T):
            x2_t, x3_t, x4_t, m5_t = x2_cnt[:, t], x3_cnt[:, t], x4_cnt[:, t], out[:, t]
            m2_t, m3_t, m4_t = self.decoder(x2_t, x3_t, x4_t, m5_t, x2_r, x3_r, x4_r)
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
    fformer_net = FFormer_Net()
    fformer_net.encoder = copy.deepcopy(seg_net.encoder)
    
    x = torch.randn(2, 4, 3, 128, 128)
    m2_out, m3_out, m4_out = fformer_net(x)
    print(m2_out.shape, m3_out.shape, m4_out.shape)