import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
#论文：CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement
#论文地址：https://arxiv.org/pdf/2209.11112v3

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.ccm = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size,
                      padding=get_padding(kernel_size), groups=inner_dim), # DepthWiseConv1d
            nn.BatchNorm1d(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ccm(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x,
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31,
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + 0.5 * self.ffm2(x)
        x = self.post_norm(x)
        return x

#Two-stage conformer (TS-Conformer)
class TSConformerBlock(nn.Module):
    def __init__(self, num_channel=64):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=num_channel,  n_head=4, ccm_kernel_size=31,
                                             ffm_dropout=0.2, attn_dropout=0.2)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':

    block = TSConformerBlock(num_channel=64)
    input = torch.randn(1, 64, 100, 80)
    output =block(input)

    print(input.size())
    print(output.size())