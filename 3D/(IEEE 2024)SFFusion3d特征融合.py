import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：A Multilevel Multimodal Fusion Transformer for Remote Sensing Semantic Segmentation
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation3D, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv3d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool3d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class SqueezeAndExciteFusionAdd3D(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd3D, self).__init__()

        self.se_1 = SqueezeAndExcitation3D(channels_in, activation=activation)
        self.se_2 = SqueezeAndExcitation3D(channels_in, activation=activation)

    def forward(self, se1, se2):
        se1 = self.se_1(se1)
        se2 = self.se_2(se2)
        out = se1 + se2
        return out

# 示例用法
if __name__ == "__main__":
    # 假设的输入数据
    input_1 = torch.randn(32, 64, 16, 128, 128)  # 输入 B C D H W
    input_2 = torch.randn(32, 64, 16, 128, 128)  # 同上

    # 打印输入数据的形状
    print(input_1.size())  # 输出: (32, 64, 16, 128, 128)
    print(input_2.size())  # 输出: (32, 64, 16, 128, 128)

    # 创建SqueezeAndExciteFusionAdd3D模块的实例
    block = SqueezeAndExciteFusionAdd3D(channels_in=64)

    # 将输入通过SqueezeAndExciteFusionAdd3D模块获得输出
    output = block(input_1, input_2)

    # 打印输出数据的形状
    print(output.size())  # 输出应该和输入形状相同: (32, 64, 16, 128, 128)
