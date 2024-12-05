import torch as th
import torch.nn as nn
#论文：Multi-Scale Temporal Frequency Convolutional Network With Axial Attention for Speech Enhancement(ICASSP 2022)
#论文地址：https://ieeexplore.ieee.org/document/9746610

class TFCM_Block(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 dila=1,
                 causal=True,
                 ):
        super(TFCM_Block, self).__init__()
        self.pconv1 = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=(1, 1)),
            nn.BatchNorm2d(cin),
            nn.PReLU(cin),
        )
        dila_pad = dila * (K[1] - 1)
        if causal:
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad, 0, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin),
                nn.PReLU(cin)
            )
        else:
            # update 22/06/21, add groups for non-casual
            self.dila_conv = nn.Sequential(
                nn.ConstantPad2d((dila_pad//2, dila_pad//2, 1, 1), 0.0),
                nn.Conv2d(cin, cin, K, 1, dilation=(1, dila), groups=cin),
                nn.BatchNorm2d(cin),
                nn.PReLU(cin)
            )
        self.pconv2 = nn.Conv2d(cin, cin, kernel_size=(1, 1))
        self.causal = causal
        self.dila_pad = dila_pad

    def forward(self, inps):
        """
            inp: B x C x F x T
        """
        outs = self.pconv1(inps)
        outs = self.dila_conv(outs)
        outs = self.pconv2(outs)
        return outs + inps


class TFCM(nn.Module):
    def __init__(self,
                 cin=24,
                 K=(3, 3),
                 tfcm_layer=6,
                 causal=True,
                 ):
        super(TFCM, self).__init__()
        self.tfcm = nn.ModuleList()
        for idx in range(tfcm_layer):
            self.tfcm.append(
                TFCM_Block(cin, K, 2**idx, causal=causal)
            )

    def forward(self, inp):
        out = inp
        for idx in range(len(self.tfcm)):
            out = self.tfcm[idx](out)
        return out





if __name__ == "__main__":
    block = TFCM(24)
    input = th.randn(2, 24, 256, 101)# B C H W
    out = block(input)
    print(out.size())