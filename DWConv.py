import torch.nn as nn
import torch
import torch.nn.functional as F
# 论文：Xception: Deep Learning with Depthwise Separable Convolutions
"""
深度可分离卷积（Depthwise Separable Convolution）是一种卷积神经网络（CNN）中的特殊类型的卷积操作，旨在减少模型参数量和计算量，从而提高模型的效率和性能。

传统的卷积操作在每个输入通道上使用一个滤波器（也称为卷积核）来进行滤波操作，这意味着每个输入通道都有自己的滤波器集合。
而深度可分离卷积将这个操作拆分成两个步骤：深度卷积和逐点卷积。

1. **深度卷积（Depthwise Convolution）**：深度卷积阶段对输入的每个通道应用一个单独的滤波器。
这意味着对于输入的每个通道，都有一个对应的滤波器集合来进行卷积操作。这一步骤不会改变输入数据的通道数。

2. **逐点卷积（Pointwise Convolution）**：逐点卷积阶段使用1x1的卷积核对深度卷积的输出进行卷积操作。
这个操作可以看作是传统的卷积操作，但是它是在每个像素点上进行的，而不是在整个图像上。逐点卷积的作用是将深度卷积的输出映射到新的特征空间，通过组合不同的通道信息来生成最终的特征图。

深度可分离卷积的优势在于它显著减少了模型的参数数量和计算量，因为它使用了更少的滤波器和更少的操作。
这使得它在资源有限的环境下更加适用，例如移动设备和嵌入式系统。同时，深度可分离卷积在一些任务上还能够提高模型的性能和泛化能力，因为它可以更好地捕获和利用特征之间的相关性。
"""

def get_dwconv(dim, kernel=7, bias=False):
    return nn.Conv2d(dim, dim, kernel_size=7, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def get_dwconv_layer2d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2,
                               groups=in_channels, bias=bias)
    point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    return nn.Sequential(depth_conv, nn.ReLU(inplace=True), point_conv, nn.ReLU(inplace=True))

def get_dwconv_layer3d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                           groups=in_channels, bias=bias)
    point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    return nn.Sequential(depth_conv, nn.ReLU(inplace=True), point_conv, nn.ReLU(inplace=True))


class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        # Depthwise convolution
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)

        # Pointwise convolution
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

if __name__ == '__main__':

    input = torch.randn(1, 3, 32, 32, 32)  # 输入 B C D H W

    # 创建 SeparableConv3d 实例
    block = SeparableConv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

    # 执行前向传播
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())