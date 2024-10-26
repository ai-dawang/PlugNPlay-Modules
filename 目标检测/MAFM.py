import torch.nn as nn
import torch
# 论文：MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603

class COI(nn.Module):
    def __init__(self, inc, k=3, p=1):
        super().__init__()
        self.outc = inc
        self.dw = nn.Conv2d(inc, self.outc, kernel_size=k, padding=p, groups=inc)
        self.conv1_1 = nn.Conv2d(inc, self.outc, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(self.outc)
        self.bn2 = nn.BatchNorm2d(self.outc)
        self.bn3 = nn.BatchNorm2d(self.outc)
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = self.bn1(x)

        x_dw = self.bn2(self.dw(x))
        x_conv1_1 = self.bn3(self.conv1_1(x))
        return self.act(shortcut + x_dw + x_conv1_1)

class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, qkv_bias=True, proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.s = nn.Linear(dim, dim, bias=qkv_bias)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1,
                                   groups=dim // self.ca_num_heads)  # kernel_size 3,5,7,9 大核dw卷积，padding 1,2,3,4
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        v = self.v(x)
        s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1,
                                                                                          2)  # num_heads,B,C,H,W
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]  # B,C,H,W
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H, W)
        s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
        self.modulator = s_out
        s_out = s_out.reshape(B, C, N).permute(0, 2, 1)
        x = s_out * v

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Multi-scale Awareness Fusion Module
class MAFM(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.outc = inc
        self.attention = MHMC(dim=inc)
        self.coi = COI(inc)
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )
        self.pre_att = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(inc * 2, inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )

    def forward(self, x, d):
        # multi = x * d
        # B, C, H, W = x.shape
        # x_cat = torch.cat((x, d, multi), dim=1)

        B, C, H, W = x.shape
        x_cat = torch.cat((x, d), dim=1)
        x_pre = self.pre_att(x_cat)
        # Attention
        x_reshape = x_pre.flatten(2).permute(0, 2, 1)  # B,C,H,W to B,N,C
        attention = self.attention(x_reshape, H, W)  # attention
        attention = attention.permute(0, 2, 1).reshape(B, C, H, W)  # B,N,C to B,C,H,W

        # COI
        x_conv = self.coi(attention)  # dw3*3,1*1,identity
        x_conv = self.pw(x_conv)  # pw

        return x_conv


if __name__ == '__main__':
    inc = 64  # 输入通道数
    block = MAFM(inc=inc)

    # 创建示例输入数据
    x = torch.randn(1, inc, 32, 32)  # B   C   H   W
    d = torch.randn(1, inc, 32, 32)  # 与 x 相同形状的深度图

    # 前向传播，计算输出
    output = block(x, d)

    # 打印输入和输出的形状
    print(f"Input x shape: {x.size()}")
    print(f"Input d shape: {d.size()}")
    print(f"Output shape: {output.size()}")