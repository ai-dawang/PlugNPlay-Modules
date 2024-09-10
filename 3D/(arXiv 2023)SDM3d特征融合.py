import torch.nn as nn
import torch.nn.functional as F
# 论文：PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# 3D图像分割即插即用模块


class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1

        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1

        kernel[:, :, 0, 2, 2] = -1
        kernel[:, :, 2, 0, 2] = -1
        kernel[:, :, 2, 2, 0] = -1

        kernel[:, :, 2, 2, 2] = 1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


class Conv3dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGNReLU, self).__init__(conv, gn, gelu)


class Conv3dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGN, self).__init__(conv, gn)
if __name__ == '__main__':
    import torch

    # 定义输入张量的形状
    input = (1, 3, 32, 32, 32)  # 输入 B C D H W

    # 创建输入张量
    input_tensor = torch.randn(input)

    # 创建引导张量
    guidance_tensor = torch.randn((1, 2, 32, 32, 32))  # 假设引导张量与输入张量大小相同

    # 创建模型
    block = SDM(in_channel=3, guidance_channels=2)

    # 将模型设置为评估模式
    block.eval()

    # 打印输入张量的形状
    print(input_tensor.size())

    # 执行前向传播
    output_tensor = block(input_tensor, guidance_tensor)

    # 打印输出张量的形状
    print(output_tensor.size())
