import torch.nn as nn
import torch
# 论文：MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603
class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class SAttention(nn.Module):
    def __init__(self, dim, sa_num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.sa_num_heads = sa_num_heads

        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        head_dim = dim // sa_num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
            self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                      N).transpose(1, 2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x.permute(0, 2, 1).reshape(B, C, H, W)

# Global Fusion Module
class GFM(nn.Module):
    def __init__(self, inc, expend_ratio=2):
        super().__init__()
        self.expend_ratio = expend_ratio
        assert expend_ratio in [2, 3], f"expend_ratio {expend_ratio} mismatch"

        self.sa = SAttention(dim=inc)
        self.dw_pw = DWPWConv(inc * expend_ratio, inc)
        self.act = nn.GELU()

    def forward(self, x, d):
        B, C, H, W = x.shape
        if self.expend_ratio == 2:
            cat = torch.cat((x, d), dim=1)
        else:
            multi = x * d
            cat = torch.cat((x, d, multi), dim=1)
        x_rc = self.dw_pw(cat).flatten(2).permute(0, 2, 1)
        x_ = self.sa(x_rc, H, W)
        x_ = x_ + x
        return self.act(x_)


if __name__ == '__main__':
    # 实例化 GFM 模块
    inc = 64  # 输入通道数
    block = GFM(inc=inc, expend_ratio=2)

    x = torch.randn(1, inc, 32, 32)  # B  C  H   W
    d = torch.randn(1, inc, 32, 32)  # 与 x 相同形状的深度图

    # 前向传播，计算输出
    output = block(x, d)

    # 打印输入和输出的形状
    print(f"Input x shape: {x.size()}")
    print(f"Input d shape: {d.size()}")
    print(f"Output shape: {output.size()}")