import torch
import torch.nn as nn
import einops
# https://ieeexplore.ieee.org/abstract/document/10632582/
# MHIAIFormer: Multi-Head Interacted and Adaptive Integrated Transformer with Spatial-Spectral Attention for Hyperspectral Image Classification, JSTARS2024
# https://github.com/Delon1364/MHIAIFormer
# Multi-Head Interacted Additive Self-Attention(MHIASA)

# Efficient Head-Interacted Additive Attention:
class EHIAAttention(nn.Module):
    def __init__(self, num_patches, dim, num_heads = 2):
        super(EHIAAttention, self).__init__()
        self.num_heads = num_heads
        self.in_dims = dim // num_heads

        # ==================添加两个linear
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)

        # w_g ->: [N, 1]
        self.w_g = nn.Parameter(torch.randn(num_patches, 1))
        self.scale_factor = num_patches ** -0.5
        self.Proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # ===================== 添加Avg分支
        self.d_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(self.in_dims, dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(num_heads, num_heads)
        self.d_avg2 = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x ->: [bs, num_patches, num_heads*in_dims]
        bs = x.shape[0]

        # ==================添加两个linear
        q = self.fc_q(x)
        x = self.fc_k(x)
        x_t = q.transpose(1, 2)

        # x_T ->: [bs, D, N]
        # x_t = x.transpose(1, 2)

        # query_weight ->: [bs, D, 1] ->: [bs, 1, D]
        query_weight = (x_t @ self.w_g).transpose(1, 2)

        A = query_weight * self.scale_factor
        A = A.softmax(dim=-1)

        # A * x_T ->: [bs, N, D]
        # G ->: [bs, D]
        G = torch.sum(A * x, dim=1)

        # ===================== 添加Avg分支
        d_avg = self.d_avg(x_t)  # [bs, D, 1]
        d_avg = torch.squeeze(d_avg, 2)  # [bs, D]
        d_avg = d_avg.reshape(bs, self.num_heads, self.in_dims)  # [bs, h, d]
        d_avg = self.gelu(self.fc(d_avg))  # [bs, h, D]
        d_avg = d_avg.reshape(bs, -1, self.num_heads)  # [bs, D, h]
        d_avg = self.fc2(d_avg)  # [bs, D, h]
        d_avg = self.sigmoid(self.d_avg2(d_avg))  # [bs, D, 1]
        d_avg = torch.squeeze(d_avg, 2)  # [bs, D]
        G = G * d_avg
        # =====================

        # G ->: [bs, N, D]
        # key.shape[1] = N
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=x.shape[1]
        )

        # out :-> [bs, N, D]
        out = self.Proj(G * x) + self.norm(x)
        # out = self.Proj(out)

        return out


if __name__ == '__main__':
    patch_size = 16
    num_patches = patch_size * patch_size
    dim = 128  # Typically dim is a multiple of num_heads

    # Instantiate the EHIAAttention
    model = EHIAAttention(num_patches=num_patches, dim=dim)

    # Create a random input tensor with shape (batch_size, num_patches, num_heads * in_dims)
    batch_size = 1
    input_tensor = torch.randn(batch_size, num_patches, dim)

    # Forward pass through the model
    output = model(input_tensor)

    # Print the shapes
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)