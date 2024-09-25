import torch
import torch.nn as nn
from einops import rearrange
#论文：ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023
#论文地址：https://arxiv.org/pdf/2303.10321
"""
摘要的翻译：
红外小目标检测（ISTD）在预警、救援和引导等方面有着广泛的应用。
然而，基于CNN的深度学习方法对缺乏清晰轮廓和纹理特征的红外小目标（IRST）分割效果不佳，而基于Transformer的方法由于缺乏卷积诱导偏差也难以取得显著的效果。
为了解决这些问题，我们提出了一种新的模型，称为双线性相关注意模型（ABC），该模型基于Transformer架构，包括一个卷积线性融合Transformer（CLFT）模块，
该模块具有用于特征提取和融合的新型注意机制，可有效增强目标特征并抑制噪声。此外，我们的模型包括一个位于网络较深层的U形卷积-扩张卷积（UCDC）模块，
它利用较深层特征的较小分辨率来获得更精细的语义信息。在公共数据集上的实验结果表明，我们的方法达到了最先进的性能。

本文的主要贡献如下：
1）基于 Transformer 结构设计的 CLFT 模块可以有效增强目标特征并抑制噪声。
2）UCDC 模块充分利用了深度特征的特点，可以更精细地处理网络的深度特征。
3）所提出的方法在所有现有的公开数据集上都达到了最佳性能。
"""
def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

#bilinear attention module (BAM)
class BAM(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(BAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        return att

class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

#dilated convolution layers(DConv)
class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        dilation = [2, 4, 2]
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x

class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = BAM(in_dim, in_feature, out_feature)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.conv(x)
        k = self.dconv(x)
        v = q + k
        att = self.att(x)
        out = torch.matmul(att, v)
        return self.gamma * out + v + x


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out

#convolution linear fusion transformer (CLFT)
class CLFT(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature):
        super(CLFT, self).__init__()
        self.attention = ConvAttention(in_dim, in_feature, out_feature)
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        x = self.attention(x)
        out = self.feedforward(x)
        return out


if __name__ == '__main__':


    block = CLFT(64,64,32*32,32) # 输入通道数，输出通道数 图像大小 H*W，H or W


    input = torch.randn(3, 64, 32, 32)   #输入tensor形状 B C H W

    # Print input shape
    print(input.size()) # 输入形状

    # Pass the input tensor through the model
    output = block(input)

    # Print output shape
    print(output.size()) # 输出形状