import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

# 论文：Locate and Verify: A Two-Stream Network for Improved Deepfake Detection
# 论文地址：https://arxiv.org/pdf/2309.11131

class CMCE(nn.Module):  # Contrastive Multimodal Contrastive Enhancement  增强模型对特征的关注度，提高模型的性能
    def __init__(self, in_channel=3):
        super(CMCE, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channel)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

    def forward(self, fa, fb):
        (b1, c1, h1, w1), (b2, c2, h2, w2) = fa.size(), fb.size()
        assert c1 == c2
        cos_sim = F.cosine_similarity(fa, fb, dim=1)
        cos_sim = cos_sim.unsqueeze(1)
        fa = fa + fb * cos_sim
        fb = fb + fa * cos_sim
        fa = self.relu(fa)
        fb = self.relu(fb)

        return fa, fb

if __name__ == '__main__':
    block = CMCE()
    fa = torch.rand(16, 3, 32, 32)
    fb = torch.rand(16, 3, 32, 32)

    fa1, fb1 = block(fa, fb)
    print(fa.size())
    print(fb.size())
    print(fa1.size())
    print(fb1.size())


class LFGA(nn.Module): # Local Feature Guidance Attention 旨在引导特征图的注意力以更好地聚焦在局部特征上
    def __init__(self, in_channel=3, out_channel=None, ratio=4):
        super(LFGA, self).__init__()
        self.chanel_in = in_channel

        if out_channel is None:
            out_channel = in_channel // ratio if in_channel // ratio > 0 else 1

        self.query_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.chanel_in)

    def forward(self, fa, fb):
        B, C, H, W = fa.size()
        proj_query = self.query_conv(fb).view(
            B, -1, H * W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(fb).view(
            B, -1, H * W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        # attention = F.normalize(energy, dim=-1)

        proj_value = self.value_conv(fa).view(
            B, -1, H * W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out + fa

        return self.relu(out)


if __name__ == '__main__':
    block = LFGA(in_channel=3, ratio=4)
    fa = torch.rand(16, 3, 32, 32)
    fb = torch.rand(16, 3, 32, 32)

    output = block(fa, fb)
    print(fa.size())
    print(fb.size())
    print(output.size())
