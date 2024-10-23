import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 论文：Efficient Long-Range Attention Network for Image Super-resolution(ECCV 2022)
# 论文地址：https://arxiv.org/pdf/2203.06697
class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y

#Local feature extraction(LFE)
class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y)
        return y

#group-wise multi-scale self-attention (GMSA)
class GMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 12], calc_attn=True):
        super(GMSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns = [channels * 2 // 3, channels * 2 // 3, channels * 2 // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=1),
                nn.BatchNorm2d(self.channels * 2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns = [channels // 3, channels // 3, channels // 3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns

#Efficient Long-range Attention Block (ELAB)
class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[2, 4, 8], shared_depth=1):
        super(ELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i + 1)] = LFE(inp_channels=inp_channels, out_channels=out_channels,
                                                      exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i + 1)] = GMSA(channels=inp_channels, shifts=shifts,
                                                         window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
        return x


if __name__ == '__main__':
    # 假设输入通道数为3（例如RGB图像），输出通道数为3
    block = ELAB(inp_channels=3, out_channels=3)
    # 创建一个示例输入张量，这里假设我们有一个批量大小为1的3通道图像，尺寸为64x64
    input = torch.randn(1, 3, 1280, 720)
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())