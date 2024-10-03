import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 论文：Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification
# 论文地址：https://ieeexplore.ieee.org/document/10098209

from einops.einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class LinAngularAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        res_kernel_size=9,
        sparse_reg=False,
    ):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim**-0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            mask = attn > 0.02 # note that the threshold could be different; adapt to your codebases.
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                self.sv_matmul(sparse, v)
                + 0.5 * v
                + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)
        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA)
    Operation where the channels are updated using a weighted sum. The weights are obtained from the (softmax
    normalized) Cross-covariance matrix (Q^T \\cdot K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.LinAngularAttention = LinAngularAttention(in_channels=128)

    def forward(self, x):
        # x1 = self.LinAngularAttention(x)
        B, N, C = x.shape                                                                                 # torch.Size([32, 784, 128])
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)    # torch.Size([3, 32, 8, 16, 784])
        q, k, v = qkv.unbind(0)                                # q, k, v = torch.Size([32, 8, 16, 784])
        q = torch.nn.functional.normalize(q, dim=-1)           # torch.Size([32, 8, 16, 784])
        k = torch.nn.functional.normalize(k, dim=-1)           # torch.Size([32, 8, 16, 784])
        attn = (q @ k.transpose(-2, -1)) * self.temperature    # torch.Size([32, 8, 16, 16])
        attn = attn.softmax(dim=-1)                            # torch.Size([32, 8, 16, 16])
        attn = self.attn_drop(attn)                            # torch.Size([32, 8, 16, 16])
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)    # torch.Size([32, 784, 128])
        x = self.proj(x)                                       # torch.Size([32, 784, 128])
        x = self.proj_drop(x)                                  # torch.Size([32, 784, 128])
        x = x # + x1
        return x


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(128, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, 128, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)

        return f1.transpose(0, 1), f2.transpose(0, 1)

class LinAngularXCA_CA(nn.Module):
    def __init__(self):
        super(LinAngularXCA_CA, self).__init__()
        self.la = LinAngularAttention(in_channels=128)
        self.xa = XCA(dim=128)
        self.cafm = CAFM()

    def forward(self, x):
        la1 = self.la(x)
        la1 = to_4d(la1, 28, 28)
        xa1 = self.xa(x)
        xa1 = to_4d(xa1, 28, 28)

        result1, result2 = self.cafm(la1, xa1)
        result = result1 + result2
        # print(result.shape)
        result = result.permute(1, 2, 0)  # 首先交换维度，变为 [32, 784, 128]

        return result

if __name__ == '__main__':
    block = LinAngularXCA_CA()
    input = torch.rand(32, 784, 128)
    output = block(input)  # 获取两个输入的输出

    print(f'Output1 shape: {output.shape}')



