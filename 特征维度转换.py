import torch
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)  # 假设输入tensor B C H W

    output = to_3d(input)
    print(output.size())    #输出shape b n c

    output1 =to_4d(output, 64, 64)  # 指定高宽 h*w =n
    print(output1.size())
