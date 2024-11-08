import torch
from torch import nn
from einops.layers.torch import Rearrange


class SpatialAttention(nn.Module):
    def __init__(self):
        """
        定义空间注意力模块，用于增强图像的空间特征。
        """
        super(SpatialAttention, self).__init__()
        # 使用1x1卷积实现空间注意力
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。

        返回：
            tensor：经过空间注意力模块处理后的张量。
        """
        # 计算平均值和最大值
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        # 经过卷积层得到空间注意力张量
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        """
        定义通道注意力模块，用于增强图像的通道特征。

        参数：
            dim (int)：输入通道数。
            reduction (int)：通道压缩比例，默认为8。
        """
        super(ChannelAttention, self).__init__()
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 通道注意力模块
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。

        返回：
            tensor：经过通道注意力模块处理后的张量。
        """
        # 经过全局平均池化
        x_gap = self.gap(x)
        # 经过通道注意力模块
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        """
        定义像素注意力模块，用于增强图像的像素特征。

        参数：
            dim (int)：输入通道数。
        """
        super(PixelAttention, self).__init__()
        # 像素注意力模块
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量。
            pattn1 (tensor)：空间注意力和通道注意力结合后的张量。

        返回：
            tensor：经过像素注意力模块处理后的张量。
        """
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        """
        定义通道、空间和像素注意力融合模块，用于融合两个张量。

        参数：
            dim (int)：输入通道数。
            reduction (int)：通道压缩比例，默认为8。
        """
        super(CGAFusion, self).__init__()
        # 实例化空间注意力、通道注意力和像素注意力模块
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        # 1x1卷积层用于输出
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        前向传播函数。

        参数：
            x (tensor)：输入张量1。
            y (tensor)：输入张量2。

        返回：
            tensor：经过通道、空间和像素注意力融合模块处理后的张量。
        """
        initial = x + y
        # 计算空间注意力和通道注意力
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        # 计算像素注意力
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        # 融合张量
        result = initial + pattn2 * x + (1 - pattn2) * y
        # 经过1x1卷积层
        result = self.conv(result)
        return result


if __name__ == '__main__':
    # 实例化通道、空间和像素注意力融合模块
    block = CGAFusion(64)
    # 创建输入张量1
    input1 = torch.rand(3, 64, 64, 64)
    # 创建输入张量2
    input2 = torch.rand(3, 64, 64, 64)
    # 前向传播
    output = block(input1, input2)
    # 输出张量大小
    print(output.size())
