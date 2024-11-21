import torch
from torch import nn
#论文：title：Gated Channel Transformation for Visual Recognition
#论文地址：https://arxiv.org/abs/1909.11519

# 定义 GCT 模块
class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            _x = torch.abs(x) if not self.after_relu else x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate


if __name__ == '__main__':


    input = torch.randn(1, 16, 32, 32)

    print(input.size())

    block = GCT(num_channels=16)

    output = block(input)

    print(output.size())