import torch
import torch.nn as nn
#论文：HALF WAVELET ATTENTION ON M-NET+ FOR LOW-LIGHT IMAGE ENHANCEMENT
#论文地址：https://arxiv.org/abs/2203.01296

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print(x_HH[:, 0, :, :])
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width])

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)


# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Half Wavelet Attention Block (HWAB)
class HWAB(nn.Module):
    def __init__(self, n_feat, o_feat, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
        super(HWAB, self).__init__()
        self.dwt = DWT()
        self.iwt = IWT()

        modules_body = \
            [
                conv(n_feat*2, n_feat, kernel_size, bias=bias),
                act,
                conv(n_feat, n_feat*2, kernel_size, bias=bias)
            ]
        self.body = nn.Sequential(*modules_body)

        self.WSA = SALayer()
        self.WCA = CALayer(n_feat*2, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)
        self.conv3x3 = nn.Conv2d(n_feat, o_feat, kernel_size=3, padding=1, bias=bias)
        self.activate = act
        self.conv1x1_final = nn.Conv2d(n_feat, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        residual = x

        # Split 2 part
        wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

        # Wavelet domain (Dual attention)
        x_dwt = self.dwt(wavelet_path_in)
        res = self.body(x_dwt)
        branch_sa = self.WSA(res)
        branch_ca = self.WCA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res) + x_dwt
        wavelet_path = self.iwt(res)

        out = torch.cat([wavelet_path, identity_path], dim=1)
        out = self.activate(self.conv3x3(out))
        out += self.conv1x1_final(residual)

        return out


if __name__ == '__main__':


    block = HWAB(n_feat=64, o_feat=64)

    input = torch.randn(1, 64, 128, 128) # B C H W

    output = block(input)

    print(input.size())
    print(output.size())