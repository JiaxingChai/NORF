import torch.nn as nn
from models.modules import conv, Fusion, RFB, SCRA
from .res2net import res2net50_v1b_26w_4s


class res2net_Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.res2net = res2net50_v1b_26w_4s(pretrained=pretrained)
        self.context2 = RFB(512, 32)
        self.context3 = RFB(1024, 32)
        self.context4 = RFB(2048, 32)
        self.fusion = Fusion(32, outchannel=32)
        self.out_conv = conv(32, 1, 1, bn=False, bias=True)
    
    def forward(self, x):
        x1, x2, x3, x4 = self.res2net(x)        # 256, 512, 1024, 2048
        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)
        a = self.fusion(x4_context, x3_context, x2_context)
        out = self.out_conv(a)
        return out, a, [x2_context, x3_context, x4_context], [x2, x3, x4]


if __name__ == '__main__':
    import torch
    x = torch.randn(2, 3, 352, 352)
    m = res2net_Encoder()
    out, a, cnt_list, x_list = m(x)
    for x_ in x_list:
        print(x_.shape)
