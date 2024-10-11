import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文：MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer
# 论文地址：https://arxiv.org/pdf/2203.13310
class DepthAwareFE(nn.Module):
    def __init__(self, output_channel_num):
        super(DepthAwareFE, self).__init__()
        self.output_channel_num = output_channel_num
        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num / 2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num / 2)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num / 2), 96, 1),
        )
        self.depth_down = nn.Conv2d(96, 12, 3, stride=1, padding=1, groups=12)
        self.acf = dfe_module(256, 256)

    def forward(self, x):
        depth = self.depth_output(x)
        N, C, H, W = x.shape
        depth_guide = F.interpolate(depth, size=x.size()[2:], mode='bilinear', align_corners=False)
        depth_guide = self.depth_down(depth_guide)
        x = x + self.acf(x, depth_guide)

        return depth, depth_guide, x


class dfe_module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(dfe_module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, feat_ffm, coarse_x):
        N, D, H, W = coarse_x.size()

        # depth prototype
        feat_ffm = self.conv1(feat_ffm)
        _, C, _, _ = feat_ffm.size()

        proj_query = coarse_x.view(N, D, -1)
        proj_key = feat_ffm.view(N, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # depth enhancement
        attention = attention.permute(0, 2, 1)
        proj_value = coarse_x.view(N, D, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(N, C, H, W)
        out = self.conv2(out)

        return out

if __name__ == '__main__':

    # 假定输入特征图的尺寸为 [N, C, H, W] = [1, 256, 64, 64]
    # 假定粗糙深度图的尺寸为 [N, D, H, W] = [1, 12, 64, 64]

    N, C, H, W = 1, 256, 64, 64
    D = 12

    # 初始化输入特征图和粗糙深度图
    feat_ffm = torch.rand(N, C, H, W)  # 输入特征图
    coarse_x = torch.rand(N, D, H, W)  # 粗糙深度图

    # 初始化dfe_module
    dfe = dfe_module(in_channels=C, out_channels=C)  # 使用相同的通道数作为示例

    # 前向传播
    output = dfe(feat_ffm, coarse_x)

    # 打印输入和输出尺寸
    print("Input feat_ffm size:", feat_ffm.size())
    print("        Output size:", output.size())
