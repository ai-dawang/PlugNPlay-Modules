import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D版本的 ChannelAttention
class ChannelAttention3D(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention3D, self).__init__()
        self.fc1 = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool3d(inputs, output_size=(1, 1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool3d(inputs, output_size=(1, 1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return x

# 3D版本的 CPCABlock
class CPCABlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super(CPCABlock3D, self).__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention3D(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv3_3_3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

        # 这里我们使用的仅有3x3x3的卷积核，可以根据需要添加更大的卷积核
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x = self.dconv3_3_3(inputs)
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

# 示例代码，打印输入输出的shape
if __name__ == "__main__":
    depth = 32  # 假设体数据的深度
    input_3d = torch.randn(1, 16, depth, 64, 64)
    print(input_3d.size())

    block_3d = CPCABlock3D(in_channels=16, out_channels=16, channelAttention_reduce=4)
    output_3d = block_3d(input_3d)
    print(output_3d.size())