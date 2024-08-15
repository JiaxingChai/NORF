import torch.nn as nn
from models.seg_net.encoders.res2net_encoder import res2net_Encoder
from models.seg_net.decoders.baseline_decoder import baseline_Decoder


class Base_Net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = res2net_Encoder(pretrained=True)
        self.decoder = baseline_Decoder()
    
    def forward(self, x):
        if len(x.shape) == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b*t, c, h, w).contiguous()
        bt, c, h, w = x.shape
        
        out, a, [x2_cnt, x3_cnt, x4_cnt], [x2, x3, x4] = self.encoder(x)
        out2, out3, out4, out5 = self.decoder(out, a, x2, x3, x4)
        
        return out2.reshape(b, t, 1, h, w), out3.reshape(b, t, 1, h, w), out4.reshape(b, t, 1, h, w), out5.reshape(b, t, 1, h, w)


if __name__ == '__main__':
    import torch
    x = torch.randn((1, 2, 3, 224, 224))
    m = Base_Net()
    out = m(x)
    for o in out:
        print(o.shape)