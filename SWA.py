import torch
import torch.nn as nn
import torch.nn.functional as F
#论文：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
#论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670
class SWA(nn.Module):
    def __init__(self, in_channels, n_heads=8, window_size=7):
        super(SWA, self).__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        padded_x = F.pad(x, [self.window_size // 2, self.window_size // 2, self.window_size // 2, self.window_size // 2], mode='reflect')

        proj_query = self.query_conv(x).view(batch_size, self.n_heads, C // self.n_heads, height * width)
        proj_key = self.key_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_key = proj_key.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)
        proj_value = self.value_conv(padded_x).unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        proj_value = proj_value.permute(0, 1, 4, 5, 2, 3).contiguous().view(batch_size, self.n_heads, C // self.n_heads, -1)

        energy = torch.matmul(proj_query.permute(0, 1, 3, 2), proj_key)
        attention = self.softmax(energy)

        out_window = torch.matmul(attention, proj_value.permute(0, 1, 3, 2))
        out_window = out_window.permute(0, 1, 3, 2).contiguous().view(batch_size, C, height, width)

        out = self.gamma * out_window + x
        return out

if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)
    block = SWA(in_channels=64)
    print(input.size())
    output = block(input)
    print(output.size())
