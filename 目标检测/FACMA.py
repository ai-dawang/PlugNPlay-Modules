import torch
from torch import nn
import math

# 论文：FCMNet: Frequency-aware cross-modality attention networks for RGB-D salient object detection
# 论文地址：https://www.sciencedirect.com/science/article/abs/pii/S0925231222003848
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i+0.5)/L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
def get_dct_weights(width,height,channel,fidx_u,fidx_v):
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i*c_part: (i+1)*c_part, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights
class FCABlock(nn.Module):

    def __init__(self, channel,width,height,fidx_u, fidx_v, reduction=16):
        super(FCABlock, self).__init__()
        mid_channel = channel // reduction
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(width,height,channel,fidx_u,fidx_v))
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x * self.pre_computed_dct_weights, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)
class SFCA(nn.Module):
    def __init__(self, in_channel,width,height,fidx_u,fidx_v):
        super(SFCA, self).__init__()

        fidx_u = [temp_u * (width // 8) for temp_u in fidx_u]
        fidx_v = [temp_v * (width // 8) for temp_v in fidx_v]
        self.FCA = FCABlock(in_channel, width, height, fidx_u, fidx_v)
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
    def forward(self, x):
        # FCA
        F_fca = self.FCA(x)
        #context attention
        con = self.conv1(x) # c,h,w -> 1,h,w
        con = self.norm(con)
        F_con = x * con
        return F_fca + F_con
class FACMA(nn.Module):
    def __init__(self,in_channel,width,height,fidx_u,fidx_v):
        super(FACMA, self).__init__()
        self.sfca_depth = SFCA(in_channel, width, height, fidx_u, fidx_v)
        self.sfca_rgb   = SFCA(in_channel, width, height, fidx_u, fidx_v)
    def forward(self, rgb, depth):
        out_d = self.sfca_depth(depth)
        out_d = rgb * out_d

        out_rgb = self.sfca_rgb(rgb)
        out_rgb = depth * out_rgb
        return out_rgb, out_d

if __name__ == '__main__':

    # 定义输入参数
    in_channel = 64
    width = 224
    height = 224
    fidx_u = [0, 1]
    fidx_v = [0, 1]

    block = FACMA(in_channel, width, height, fidx_u, fidx_v)

    # 假设的RGB和深度输入
    rgb_input = torch.randn(1, in_channel, width, height)  # Batch size为1
    depth_input = torch.randn(1, in_channel, width, height)  # Batch size为1

    # 通过FACMA
    out_rgb, out_d = block(rgb_input, depth_input)

    # 打印输入输出形状
    print("RGB 输入形状:", rgb_input.shape)
    print("深度 输入形状:", depth_input.shape)
    print("RGB 输出形状:", out_rgb.shape)
    print("深度 输出形状:", out_d.shape)