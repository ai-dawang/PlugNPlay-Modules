import torch.nn as nn
import torch
#论文：DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement
#论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0925231224003564
class Inc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Inc, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(3, 3), stride=(1, 1), dilation=1,
                      padding=(3 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1,
                      padding=(1 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(5, 5), stride=(1, 1), dilation=1,
                      padding=(5 - 1) // 2),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),

        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=(1, 1), stride=(1, 1), dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        return torch.cat([o1, o2, o3, o4], dim=1)


class Flatten(nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign()
        )

    def forward(self, input):
        return input * self.module(input).unsqueeze(2).unsqueeze(3).expand_as(input)


class DICAM(nn.Module):
    def __init__(self):
        super(DICAM, self).__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)

        self.layer_2_r = CAM(256, 4)
        self.layer_2_g = CAM(256, 4)
        self.layer_2_b = CAM(256, 4)

        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)

        self.layer_tail = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=24, kernel_size=(3, 3), stride=(1, 1), padding=(3 - 1) // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=(1, 1), stride=(1, 1), padding=(1 - 1) // 2),
            nn.Sigmoid()

        )

    def forward(self, input):
        input_r = torch.unsqueeze(input[:, 0, :, :], dim=1)
        input_g = torch.unsqueeze(input[:, 1, :, :], dim=1)
        input_b = torch.unsqueeze(input[:, 2, :, :], dim=1)

        layer_1_r = self.layer_1_r(input_r)
        layer_1_g = self.layer_1_g(input_g)
        layer_1_b = self.layer_1_b(input_b)

        layer_2_r = self.layer_2_r(layer_1_r)
        layer_2_g = self.layer_2_g(layer_1_g)
        layer_2_b = self.layer_2_b(layer_1_b)

        layer_concat = torch.cat([layer_2_r, layer_2_g, layer_2_b], dim=1)

        layer_3 = self.layer_3(layer_concat)
        layer_4 = self.layer_4(layer_3)

        output = self.layer_tail(layer_4)
        return output


if __name__ == '__main__':

    block = DICAM()
    input = torch.randn(1, 3, 128, 128)
    output = block(input)
    print(input.size())
    print(output.size())