import torch
import torch.nn as nn
# 论文：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
class WCMF(nn.Module):
    def __init__(self,channel=256):
        super(WCMF, self).__init__()
        self.conv_r1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_d1 = nn.Sequential(nn.Conv2d(channel, channel, 1, 1, 0), nn.BatchNorm2d(channel), nn.ReLU())

        self.conv_c1 = nn.Sequential(nn.Conv2d(2*channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def fusion(self,f1,f2,f_vec):

        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2
    def forward(self,rgb,depth):
        Fr = self.conv_r1(rgb)
        Fd = self.conv_d1(depth)
        f = torch.cat([Fr, Fd],dim=1)
        f = self.conv_c1(f)
        f = self.conv_c2(f)
        # f = self.avgpool(f)
        Fo = self.fusion(Fr, Fd, f)
        return Fo


if __name__ == '__main__':

    block = WCMF(channel=256)

    # 创建RGB和深度输入的假设张量
    rgb_input = torch.randn(1, 256, 224, 224)
    depth_input = torch.randn(1, 256, 224, 224)

    # 通过WCMF模型
    output = block(rgb_input, depth_input)

    # 打印输入和输出的shape
    print(rgb_input.size())
    print(depth_input.size())
    print(output.size())