import torch
import torch.nn as nn
#论文：CLEEGN: A Convolutional Neural Network for Plug-and-Play Automatic EEG Reconstruction
#论文地址：https://arxiv.org/pdf/2210.05988v2.pdf

class Permute2d(nn.Module):
    def __init__(self, shape):
        super(Permute2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)


class CLEEGN(nn.Module):
    def __init__(self, n_chan, fs, N_F=20, tem_kernelLen=0.1):
        super(CLEEGN, self).__init__()
        self.n_chan = n_chan
        self.N_F = N_F
        self.fs = fs
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_chan, (n_chan, 1), padding="valid", bias=True),
            Permute2d((0, 2, 1, 3)),
            nn.BatchNorm2d(1, eps=1e-3, momentum=0.99)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(N_F, N_F, (1, int(fs * tem_kernelLen)), padding="same", bias=True),
            nn.BatchNorm2d(N_F, eps=1e-3, momentum=0.99)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(N_F, n_chan, (n_chan, 1), padding="same", bias=True),
            nn.BatchNorm2d(n_chan, eps=1e-3, momentum=0.99)
        )
        self.conv5 = nn.Conv2d(n_chan, 1, (n_chan, 1), padding="same", bias=True)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        # decoder
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.conv5(x)
        return x


if __name__ == '__main__':

    # 定义输入张量的参数
    batch_size = 1  # 批次大小，表示处理一个样本
    n_channels = 56  # EEG信号的通道数
    sampling_rate = 128.0  # 信号采样频率，单位为Hz
    time_length = int(sampling_rate)  # 时间长度（宽度），即一个时间序列周期内的数据点数

    # 初始化模型
    model = CLEEGN(n_chan=n_channels, fs=sampling_rate, N_F=20, tem_kernelLen=0.1)

    # 生成随机输入张量，模拟EEG数据
    input_tensor = torch.randn(batch_size, 1, n_channels, time_length)  # (batch_size, channels, height, width)

    # 执行前向传播
    output = model(input_tensor)

    # 输出输入和输出张量的形状
    print(f'输入张量形状: {input_tensor.shape}')
    print(f'输出张量形状: {output.shape}')

