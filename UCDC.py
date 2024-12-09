import torch.nn as nn
import torch.utils.data
import torch
#论文：ABC: Attention with Bilinear Correlation for Infrared Small Target Detection ICME2023
#论文地址：https://arxiv.org/pdf/2303.10321

def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


#u-shaped convolution-dilated convolution (UCDC)
class UCDC(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UCDC, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out


if __name__ == '__main__':

    block = UCDC(64, 64)


    input = torch.randn(1, 64, 32, 32)

    print(input.size())

    output = block(input)

    print(output.size())
