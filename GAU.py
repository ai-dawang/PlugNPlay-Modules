import torch
import torch.nn as nn

class TA(nn.Module):
    def __init__(self,  T,ratio=2):

        super(TA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(T, T // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(T // ratio, T, 1,bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        # B,T,C
        out1 = self.sharedMLP(avg)
        max = self.max_pool(x)
        # B,T,C
        out2 = self.sharedMLP(max)
        out = out1+out2

        return out

# task classifictaion or generation
class SCA(nn.Module):
    def __init__(self, in_planes, kerenel_size,ratio = 1):
        super(SCA, self).__init__()
        self.sharedMLP = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // ratio, kerenel_size, padding='same', bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, kerenel_size, padding='same', bias=False),)
    def forward(self, x):
        b,t, c, h, w = x.shape
        x = x.flatten(0,1)
        x = self.sharedMLP(x)
        out = x.reshape(b,t, c, h, w)
        return out
if __name__ == '__main__':

    block1 = TA(T=10)  # 假设输入有10个时间步长
    print("TA模型结构：\n", block1)

    # 创建SCA模型
    block2 = SCA(in_planes=64, kerenel_size=3)  # 假设输入通道数为64
    print("\nSCA模型结构：\n", block2)

    # 创建随机输入数据
    batch_size = 4
    time_steps = 10
    channels = 64
    height = 32
    width = 32
    input = torch.randn(batch_size, time_steps, channels, height, width)
    print("\n输入数据形状：", input.size())

    # 测试TA模型
    output = block1(input)
    print("TA模型输出形状：", output.shape)

    # 测试SCA模型
    output2 = block2(input)
    print("SCA模型输出形状：", output2.shape)