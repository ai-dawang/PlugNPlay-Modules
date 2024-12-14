import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：A dual encoder crack segmentation network with Haar wavelet-based high-low frequency attention
# 论文地址：https://doi.org/10.1016/j.eswa.2024.124950


class PositionEmbedding(nn.Module):
    def __init__(self, t=10000):
        super().__init__()
        self.t = t

    def forward(self, x):
        B, N, C = x.shape
        assert C % 2 == 0, 'dim must be divided 2'

        pos_embed = torch.zeros(N, C, dtype=torch.float32)

        N_num = torch.arange(N, dtype=torch.float32)

        o = torch.arange(C//2, dtype=torch.float32)
        o /= C/2.
        o = 1. / (self.t**o)

        out = N_num[:, None] @ o[None, :]

        sin_embed = torch.sin(out)
        cos_embed = torch.cos(out)

        pos_embed[:, 0::2] = sin_embed
        pos_embed[:, 1::2] = cos_embed

        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)
        return pos_embed

class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class FCHiLo1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim
        self.pos = PositionEmbedding()

        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.ws != 1:
            # self.wt = DWTForward(J=1, mode='zero', wave='haar')
            self.wt = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        else:
            self.sr = nn.Sequential()

        if self.l_heads > 0:
            self.l_q = DSC(self.dim, self.l_dim)
            self.l_kv = DSC(self.dim, self.l_dim*2)
            self.l_proj = DSC(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = DSC(self.dim, self.h_dim*3)
            self.h_proj = DSC(self.h_dim, self.h_dim)

    def hi_lofi(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.ws != 1:
            # low_feats, yH = self.wt(x)
            low_feats = self.wt(x)
        else:
            low_feats = self.sr(x)

        high_feats = F.interpolate(low_feats, size=H, mode='nearest')
        high_feats = high_feats - x

        if self.l_heads!=0:
            l_q = self.l_q(x).permute(0, 2, 3, 1).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
            if self.ws > 1:
                l_kv = self.l_kv(low_feats).permute(0, 2, 3, 1).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            else:
                l_kv = self.l_kv(x).permute(0, 2, 3, 1).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            l_k, l_v = l_kv[0], l_kv[1]

            l_attn = (l_q @ l_k.transpose(-2, -1)) * self.scale
            l_attn = l_attn.softmax(dim=-1)

            l_x = (l_attn @ l_v).transpose(1, 2).reshape(B, H, W, self.l_dim).permute(0, 3, 1, 2)
            l_x = self.l_proj(l_x).permute(0, 2, 3, 1)


        if self.h_heads!=0:
            h_group, w_group = H // self.ws, W // self.ws
            total_groups = h_group * w_group
            h_qkv = self.h_qkv(high_feats).permute(0, 2, 3, 1).\
                reshape(B, h_group, self.ws, w_group, self.ws, 3*self.h_dim).\
                transpose(2, 3).reshape(B, total_groups, -1, 3, self.h_heads,
                                        self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
            h_q, h_k, h_v = h_qkv[0], h_qkv[1], h_qkv[2]

            h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale
            h_attn = h_attn.softmax(dim=-1)
            h_attn = (h_attn @ h_v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
            h_x = h_attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim).permute(0, 3, 1, 2)

            h_x = self.h_proj(h_x).permute(0, 2, 3, 1)


        if self.h_heads!=0 and self.l_heads!=0:
            out = torch.cat([l_x, h_x], dim=-1)
            out = out.reshape(B, N, C)

        if self.l_heads==0:
            out = h_x.reshape(B, N, C)

        if self.h_heads==0:
            out = l_x.reshape(B, N, C)

        return out

    def forward(self, x):
        return self.hi_lofi(x)

class FFN1(nn.Module):
    def __init__(self, dim, h_dim=None, out_dim=None):
        super().__init__()
        self.h_dim = dim*2 if h_dim==None else h_dim
        self.out_dim = dim if out_dim==None else out_dim

        self.act = nn.GELU()
        self.fc1 = DSC(dim, self.h_dim)
        self.norm = nn.BatchNorm2d(self.out_dim)
        self.fc2 = DSC(self.h_dim, self.h_dim)
        self.fc3 = IDSC(self.h_dim, self.out_dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.act(self.fc3(self.act(self.fc2(self.act(self.fc1(x))))))
        x = self.norm(x).reshape(B, C, -1).permute(0, 2, 1)

        return x

class Block1(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=2, alpha=0.5, qkv_bias=False, qk_scale=None, h_dim=None, out_dim=None):
        super().__init__()
        self.hilo = FCHiLo1(dim, num_heads, qkv_bias, qk_scale, window_size, alpha)
        self.ffn = FFN1(dim, h_dim, out_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = x + self.norm1(self.hilo(x))
        x = x + self.norm2(self.ffn(x))
        return x


if __name__ == '__main__':

    input = torch.randn(1, 1024, 64)  # B N C

    block1 = Block1(64)
    print(input.size())
    output_block1 = block1(input)
    print(output_block1.size())

    ffn1 = FFN1(64)
    print(input.size())
    output_ffn1 = ffn1(input)
    print(output_ffn1.size())

    # Instantiate FCHiLo1
    fchilo1 = FCHiLo1(64)
    print(input.size())
    output_fchilo1 = fchilo1(input)
    print(output_fchilo1.size())