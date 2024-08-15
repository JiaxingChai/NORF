import torch
import torch.nn as nn
from models.mem_net.resnet import resnet18
from models.modules import RFB
from models.mem_net.cbam import CBAM


class Value_Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        network = resnet18(pretrained=True, extra_dim=1)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256
        self.layer4 = network.layer4 # 1/32, 512
        
        self.rfb = RFB(512, 32)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1) 
        self.cbam = CBAM(32, reduction_ratio=4)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
    
    def forward(self, image_feat32_32, image, mask):
        """
        image: b 3 h w, reference image
        mask: b 1 h w, mask for the reference image
        """
        # print(image_feat32_32.shape, image.shape, mask.shape)
        g = torch.cat([image, mask], dim=1)
        g = self.conv1(g)
        g = self.bn1(g)
        g = self.maxpool(g)
        g = self.relu(g)
        
        g = self.layer1(g)
        g = self.layer2(g)
        g = self.layer3(g)
        g = self.layer4(g)
        g = self.rfb(g)
        
        # fusion
        g = torch.cat([g, image_feat32_32], dim=1)
        g = self.conv2(g)
        g = self.cbam(g)
        g = self.conv3(g)    
        return g


class QKV_Encoder(nn.Module):
    def __init__(self, in_chs, proj_chs):
        super().__init__()
        self.qk_proj = nn.Conv2d(in_chs, proj_chs, kernel_size=3, padding=1)
        self.key_scale = nn.Conv2d(in_chs, 1, kernel_size=3, padding=1)
        self.query_select = nn.Conv2d(in_chs, proj_chs, kernel_size=3, padding=1)
        self.v_proj = Value_Encoder()
    
    def forward(self, f32_32, image, mask):
        """
        f32_32: torch.Tensor, shape=(b, T, 32, h, w)
        image: b 3 h w, reference image
        mask: b 1 h w, mask for the reference image
        """
        b, T, c, h, w = f32_32.shape
        value = self.v_proj(f32_32[:, 0], image, mask)
        
        f32_32 = f32_32.reshape(b * T, c, h, w)
        qk_proj = self.qk_proj(f32_32)
        key_scale = self.key_scale(f32_32)**2 + 1
        query_select = torch.sigmoid(self.query_select(f32_32))
        
        # b*T c h w -> b T c h w
        qk_proj = qk_proj.view(b, T, *qk_proj.shape[-3:]).contiguous()
        key_scale = key_scale.view(b, T, *key_scale.shape[-3:]).contiguous()
        query_select = query_select.view(b, T, *query_select.shape[-3:]).contiguous()
        
        return qk_proj, key_scale, query_select, value


if __name__ == '__main__':
    qkv_proj = QKV_Encoder(32, 32)
    f32_32 = torch.randn(8, 3, 32, 11, 11)
    image = torch.randn(8, 3, 352, 352)
    mask = torch.randn(8, 1, 352, 352)
    qk_proj, key_scale, query_select, v = qkv_proj(f32_32, image, mask)
    print(qk_proj.shape, key_scale.shape, query_select.shape, v.shape)

