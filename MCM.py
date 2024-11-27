import torch.nn as nn
import torch
import torch.nn.functional as F
# 论文：MAGNet: Multi-scale Awareness and Global fusion Network for RGB-D salient object detection | KBS
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0950705124007603
# github地址：https://github.com/mingyu6346/MAGNet

TRAIN_SIZE = 384

class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x1, x2):
        x2_upsample = self.upsample2(x2)  # 上采样
        x2_rc = self.rc(x2_upsample)  # 减少通道数
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 减少通道数2
        x_forward = x_forward + shortcut
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward


if __name__ == '__main__':

    inc = 64  # 输入通道数
    outc = 32  # 输出通道数
    mcm = MCM(inc=inc, outc=outc)

    x1 = torch.randn(1, outc, 96, 96)  # Batch size=1, Channels=outc, Height=96, Width=96
    x2 = torch.randn(1, inc, 48, 48)  # Batch size=1, Channels=inc, Height=48, Width=48

    pred, x_forward = mcm(x1, x2)

    print(x1.size())
    print(x2.size())
    print(pred.size())
    print(x_forward.size())