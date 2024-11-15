import torch as th
import torch.nn as nn
#论文：Multi-Scale Temporal Frequency Convolutional Network With Axial Attention for Speech Enhancement (ICASSP 2022)
#论文地址：https://ieeexplore.ieee.org/document/9746610

class FD(nn.Module):
    def __init__(self, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0)):
        super(FD, self).__init__()
        self.fd = nn.Sequential(
            nn.Conv2d(cin, cout, K, S, P, groups=2),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout)
        )

    def forward(self, x):
        return self.fd(x)


class FU(nn.Module):
    def __init__(self, cin, cout, K=(7, 1), S=(4, 1), P=(2, 0), O=(1, 0)):
        super(FU, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin*2, cin, (1, 1)),
            nn.BatchNorm2d(cin),
            nn.Tanh(),
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 1)),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout),
        )
        #  22/06/13 update, add groups = 2
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(cout, cout, K, S, P, O, groups=2),
            nn.BatchNorm2d(cout),
            nn.PReLU(cout)
        )

    def forward(self, fu, fd):
        """
        fu, fd: B C F T
        """
        outs = self.pconv1(th.cat([fu, fd], dim=1))*fd
        outs = self.pconv2(outs)
        outs = self.conv3(outs)
        return outs


def test_fd():
    net = FD(4, 8)
    inps = th.randn(3, 4, 256, 101)
    print(net(inps).shape)


if __name__ == "__main__":
    test_fd()