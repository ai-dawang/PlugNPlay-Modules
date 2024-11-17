import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

# 论文地址：https://arxiv.org/pdf/2405.01992
# 论文：SFFNet: A Wavelet-Based Spatial and Frequency Domain Fusion Network for Remote Sensing Segmentation, arxiv2405

class Bconv(nn.Module):
    def __init__(self, ch_in, ch_out, k, s):
        '''
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        :param k: 卷积核尺寸
        :param s: 步长
        :return:
        '''
        super(Bconv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, padding=k // 2)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU()

    def forward(self, x):
        '''
        :param x: 输入
        :return:
        '''
        return self.act(self.bn(self.conv(x)))


class SppCSPC(nn.Module):
    def __init__(self, ch_in, ch_out):
        '''
        :param ch_in: 输入通道
        :param ch_out: 输出通道
        '''
        super(SppCSPC, self).__init__()
        # 分支一
        self.conv1 = nn.Sequential(
            Bconv(ch_in, ch_out, 1, 1),
            Bconv(ch_out, ch_out, 3, 1),
            Bconv(ch_out, ch_out, 1, 1)
        )
        # 分支二（SPP）
        self.mp1 = nn.MaxPool2d(5, 1, 5 // 2)  # 卷积核为5的池化
        self.mp2 = nn.MaxPool2d(9, 1, 9 // 2)  # 卷积核为9的池化
        self.mp3 = nn.MaxPool2d(13, 1, 13 // 2)  # 卷积核为13的池化

        # concat之后的卷积
        self.conv1_2 = nn.Sequential(
            Bconv(4 * ch_out, ch_out, 1, 1),
            Bconv(ch_out, ch_out, 3, 1)
        )

        # 分支三
        self.conv3 = Bconv(ch_in, ch_out, 1, 1)

        # 此模块最后一层卷积
        self.conv4 = Bconv(2 * ch_out, ch_out, 1, 1)

    def forward(self, x):
        # 分支一输出
        output1 = self.conv1(x)

        # 分支二池化层的各个输出
        mp_output1 = self.mp1(output1)
        mp_output2 = self.mp2(output1)
        mp_output3 = self.mp3(output1)

        # 合并以上并进行卷积
        result1 = self.conv1_2(torch.cat((output1, mp_output1, mp_output2, mp_output3), dim=1))

        # 分支三
        result2 = self.conv3(x)

        return self.conv4(torch.cat((result1, result2), dim=1))

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GlobalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.Conv2d(dim,dim,kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.Conv2d(dim,dim,kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class LocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 window_size=8,
                 ):
        super().__init__()

        self.local = SppCSPC(dim,dim)
        # self.bam = BAM(gate_channel=dim)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local(x)

        out = self.pad_out(local)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class LocalBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn =LocalAttention(dim,window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class multilocalBlock(nn.Module):
    expansion = 1
    def __init__(self,dim=256,outdim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.down = Conv(dim,outdim,kernel_size=3,stride=2,dilation=1,bias=False)
        self.norm1 = norm_layer(outdim)
        self.attn =LocalAttention(outdim,window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=outdim, hidden_features=mlp_hidden_dim, out_features=outdim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(outdim)

    def forward(self, x):
        x = self.down(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.drop_path(self.norm2(x))

        return x
class GlobalBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
class GlBlock(nn.Module):
    expansion = 1
    def __init__(self, dim=256,outdim = 256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8,C=0,H=0,W=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.down = Conv(dim, outdim, kernel_size=3, stride=2, dilation=1, bias=False)
    def forward(self, x):
        x = self.down(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.norm2(x)

        return x

# feature mapping stage(FMS)
class FMS(nn.Module):
    def __init__(self, in_ch, out_ch,num_heads=8, window_size=8):
        super(FMS, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.glb = GlBlock(dim=in_ch,outdim=in_ch,num_heads=num_heads, window_size=window_size)
        self.localb=multilocalBlock(dim=in_ch,outdim=in_ch,num_heads=8, window_size=window_size)
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x,imagename=None):

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)


        glb = self.outconv_bn_relu_glb(self.glb(x))
        local = self.outconv_bn_relu_local(self.localb(x))
        return yL,yH,glb,local


if __name__ == '__main__':

    block = FMS(in_ch=64, out_ch=128)


    input = torch.randn(1, 64, 256, 256)

    # 前向传播
    yL, yH, glb, local = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(yL.size())
    print(yH.size())
    print(glb.size())
    print(local.size())