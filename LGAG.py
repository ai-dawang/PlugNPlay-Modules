import torch.nn as nn
import torch
# 论文：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# 论文地址：https://arxiv.org/pdf/2405.06880


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int=16, kernel_size=3, groups=1, activation='relu'):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


if __name__ == '__main__':
    # 示例输入
    g = torch.randn(1, 32, 64, 64)
    x = torch.randn(1, 64, 64, 64)

    # 实例化LGAG
    lgag = LGAG(F_g=32, F_l=64)

    # 打印输入的shape
    print("输入 g 的 shape:", g.shape)
    print("输入 x 的 shape:", x.shape)

    # 前向传播并打印输出的shape
    output = lgag(g, x)
    print("输出的 shape:", output.shape)