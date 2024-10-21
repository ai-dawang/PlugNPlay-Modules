import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
#论文：GraphFit: Learning Multi-scale Graph-Convolutional Representation for Point Cloud Normal Estimation
#论文地址：https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136920646.pdf

# 计算点云中每个点的 k 个最近邻居的函数
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 计算内积
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 平方和
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 计算成对距离

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # 获取 k 个最近邻居的索引
    return idx

# 从点云中构建图特征的函数
def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # 如果未提供，计算 k-NN
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # 返回图特征

# 用于编码邻居点之间关系的图块
class GraphBlock(nn.Module):
    def __init__(self, dim=64, k1=40, k2=20):
        super(GraphBlock, self).__init__()
        self.dim = dim
        self.k1 = k1
        self.k2 = k2

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn4 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm2d(self.dim)

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x_knn1 = get_graph_feature(x, k=self.k1)  # 获取 k1 邻居的 k-NN 图特征
        x_knn1 = self.conv1(x_knn1)  # 应用第一个卷积
        x_knn1 = self.conv2(x_knn1)  # 应用第二个卷积
        x_k1 = x_knn1.max(dim=-1, keepdim=False)[0]  # 最大池化

        x_knn2 = get_graph_feature(x, self.k2)  # 获取 k2 邻居的 k-NN 图特征
        x_knn2 = self.conv3(x_knn2)  # 应用第三个卷积
        x_knn2 = self.conv4(x_knn2)  # 应用第四个卷积
        x_k1 = x_k1.unsqueeze(-1).repeat(1, 1, 1, self.k2)  # 为拼接操作扩展维度并重复

        out = torch.cat([x_knn2, x_k1], dim=1)  # 拼接特征
        out = self.conv5(out)  # 应用第五个卷积
        out = out.max(dim=-1, keepdim=False)[0]  # 最大池化

        return out  # 返回输出特征


if __name__ == '__main__':

    block = GraphBlock(dim=64, k1=40, k2=20)
    batch_size = 1
    num_dims = 64
    num_points = 1024
    input = torch.randn(batch_size, num_dims, num_points).cuda()
    block = block.cuda()
    # 进行前向传播
    output = block(input)

    # 打印输入和输出的形状
    print(input.size())
    print(output.size())