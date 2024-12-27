from torch import nn
import torch
from einops import rearrange

# 论文题目：MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
# 论文链接：https://arxiv.org/pdf/2110.02178


# 预定义一个带有层归一化的预处理模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)  # 层归一化，标准化输入
        self.fn = fn  # 用于传入的函数（例如 Attention 或 FeedForward）

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)  # 对归一化后的输入应用函数


# 定义一个前馈神经网络模块，用于 MLP 层
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # 线性层，输入维度到 MLP 维度
            nn.SiLU(),  # SiLU 激活函数
            nn.Dropout(dropout),  # Dropout，防止过拟合
            nn.Linear(mlp_dim, dim),  # 线性层，将 MLP 维度还原为输入维度
            nn.Dropout(dropout)  # Dropout
        )

    def forward(self, x):
        return self.net(x)  # 输出前馈网络的结果


# 定义注意力模块，用于计算多头自注意力
class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim  # 内部维度为头数乘以每头的维度
        project_out = not (heads == 1 and head_dim == dim)  # 判断是否需要输出投影

        self.heads = heads  # 注意力头的数量
        self.scale = head_dim ** -0.5  # 缩放因子，用于稳定训练

        self.attend = nn.Softmax(dim=-1)  # 使用 Softmax 计算注意力权重
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 线性变换生成查询、键、值

        # 输出层，如果没有单独的投影层则直接使用 Identity
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 将查询、键和值分成三个部分
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)  # 重排维度
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 计算注意力分数
        attn = self.attend(dots)  # 对注意力分数应用 Softmax
        out = torch.matmul(attn, v)  # 根据注意力权重加权值向量
        out = rearrange(out, 'b p h n d -> b p n (h d)')  # 重排回原始维度
        return self.to_out(out)  # 返回投影输出


# Transformer 模块，由多层注意力和前馈网络组成
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 初始化层列表
        for _ in range(depth):  # 根据深度循环添加层
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),  # 预归一化注意力模块
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))  # 预归一化前馈模块
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:  # 遍历注意力和前馈网络层
            out = out + att(out)  # 残差连接，应用注意力
            out = out + ffn(out)  # 残差连接，应用前馈网络
        return out


# MobileViT 的注意力模块，结合了局部和全局表示
class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size  # 设置 patch 的高度和宽度
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 局部卷积
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)  # 用于通道变换的 1x1 卷积

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)  # Transformer 模块用于全局表示

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)  # 将维度变换回原通道
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)  # 用于融合的卷积层

    def forward(self, x):
        y = x.clone()  # 复制输入张量 y = x 以保留局部特征

        ## 局部表示
        y = self.conv2(self.conv1(x))  # 使用卷积层获得局部特征

        ## 全局表示
        _, _, h, w = y.shape  # 获取 y 的高度和宽度
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # 重排为 patch 格式
        y = self.trans(y)  # 应用 Transformer 进行全局特征提取
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # 恢复为原始形状

        ## 融合
        y = self.conv3(y)  # 维度变换回原通道
        y = torch.cat([x, y], 1)  # 拼接局部和全局特征
        y = self.conv4(y)  # 融合后的卷积操作

        return y  # 返回融合结果

if __name__ == '__main__':
    m = MobileViTAttention(in_channel=512)
    input = torch.randn(1, 512, 49, 49)  # 生成输入张量，大小为 (1, 512, 49, 49)
    output = m(input)  # 应用 MobileViTAttention 模块
    print(input.shape)  # 打印输入张量的形状
    print(output.shape)  # 打印输出张量的形状
