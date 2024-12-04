import torch
import torch.nn as nn
#论文：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
#论文：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * torch.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out


class PCBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(PCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, ratio)
        self.spatial_attention = SpatialAttentionModule()
        self.position_attention = PAM(in_channels)

    def forward(self, x):
        x_c = self.channel_attention(x)
        x_s = self.spatial_attention(x_c)
        x_p = self.position_attention(x)
        out = x_s + x_p
        return out


if __name__ == '__main__':

    input = torch.randn(1, 64,32, 32)
    block = PCBAM(in_channels=64)
    print(input.size())
    output = block(input)
    print(output.size())







