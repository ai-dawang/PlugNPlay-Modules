import torch
import torch.nn as nn
from einops import rearrange
# 论文地址：https://arxiv.org/pdf/2408.14087
# 论文：LSM-YOLO: A Compact and Effective ROI Detector for Medical Detection


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class LAE(nn.Module):
    # Light-weight Adaptive Extraction
    def __init__(self, ch, group=16) -> None:
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )

        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)

        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x


if __name__ == '__main__':

    input = torch.randn(1, 16, 64, 64) # B C H W
    block = LAE(ch=16)
    output = block(input)
    print(input.size())
    print(output.size())