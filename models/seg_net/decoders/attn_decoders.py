import torch.nn as nn
from models.modules import SCRA
from torch.nn import functional as F


class att_Decoder(nn.Module):
    def __init__(self, out_channel=1, k=8):
        super().__init__()
        self.attention2 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)
        self.attention3 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)
        self.attention4 = SCRA(32, 32, channel=32, t=3, depth=2, kernel_size=3, k=k)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, out, fusion, x4, x3, x2, h, w):
        base_size = (h, w)
        a5 = out
        out5 = self.res(a5, base_size)  # F.interpolate(a5, size=base_size, mode='bilinear', align_corners=False)

        f4, a4, corrtcted_map4 = self.attention4(x4, self.ret(fusion, x4), a5)
        out4 = self.res(a4, base_size)

        f3, a3, corrtcted_map3 = self.attention3(x3, self.ret(x4, x3), a4)
        out3 = self.res(a3, base_size)

        _, a2, corrtcted_map2 = self.attention2(x2, self.ret(x3, x2), a3)
        out2 = self.res(a2, base_size)

        corrtcted_map2 = self.res(corrtcted_map2, base_size)
        corrtcted_map3 = self.res(corrtcted_map3, base_size)
        corrtcted_map4 = self.res(corrtcted_map4, base_size)
        return [out2, out3, out4, out5], [corrtcted_map2, corrtcted_map3, corrtcted_map4]