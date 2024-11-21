import torch
from torch import nn
import torch.nn.functional as F

#可以缝合在跳跃连接部分
class LayerNorm(nn.Module):
    """ From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)"""

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


class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + 1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x
if __name__ == '__main__':
    # 创建模拟输入数据
    xh = torch.randn(1, 64, 32, 32)  # 输入 xh 的形状为 [B C H W]
    xl = torch.randn(1, 64, 16, 16)  # 输入 xl 的形状为 [B C H/2 W2]
    mask = torch.randn(1, 1, 16, 16)  # 蒙版张量的形状为 [B 1 H/2 W/2]
    # 实例化模块
    block = group_aggregation_bridge(dim_xh=64, dim_xl=64)

    # 打印输入的形状
    print("输入 xh 的形状:", xh.size())
    print("输入 xl 的形状:", xl.size())
    print("蒙版张量的形状:", mask.size())

    # 进行前向传播
    output = block(xh, xl, mask)

    # 打印输出的形状
    print("输出的形状:", output.size())