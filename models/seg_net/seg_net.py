import torch.nn as nn
from models.seg_net.encoders.res2net_encoder import res2net_Encoder
from models.seg_net.decoders.attn_decoders import att_Decoder


class Seg_Net(nn.Module):
    def __init__(self,):
        """"""
        super().__init__()
        self.encoder = res2net_Encoder(pretrained=True)
        self.decoder = att_Decoder()
    
    def forward(self, x):
        if len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b*t, c, h, w).contiguous()
        bt, c, h, w = x.shape
        
        out, a, [x2_cnt, x3_cnt, x4_cnt], [x2, x3, x4] = self.encoder(x)
        [out2, out3, out4, out5], [cor_map2, cor_map3, cor_map4] = self.decoder(out, a, x4_cnt, x3_cnt, x2_cnt, h, w)
        
        return [out2.reshape(b, t, 1, h, w), out3.reshape(b, t, 1, h, w), out4.reshape(b, t, 1, h, w), out5.reshape(b, t, 1, h, w)], \
               [cor_map2.reshape(b, t, 1, h, w), cor_map3.reshape(b, t, 1, h, w), cor_map4.reshape(b, t, 1, h, w)]


if __name__ == '__main__':
    import torch
    x = torch.randn(1, 2, 3, 352, 352)
    m = Seg_Net()
    out, cor = m(x)
    for x_ in x:
        print(x_.shape)