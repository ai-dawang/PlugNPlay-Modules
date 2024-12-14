import math
import torch
from torch import nn
from functools import partial
from einops import repeat
import torch.nn.functional as F
from timm.models.layers import DropPath
#论文：Perspective+ Unet: Enhancing Segmentation with Bi-Path Fusion and Efficient Non-Local Attention for Superior Receptive Fields [MICCAI2024]
#论文地址：https://arxiv.org/abs/2406.14052

def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    some = True
    q, r = torch.linalg.qr(unstructured_block.cpu(), 'reduced' if some else 'complete')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=False, eps=1e-4, device=None):
    b, h, *_ = data.shape

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0)
    diag_data = diag_data.unsqueeze(dim=-1)

    data_dash = ratio * (torch.exp(data_dash - diag_data) + eps)

    return data_dash.type_as(data)


# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

#Efficient Non-Local Attention Mechanism (ENLA)
class ENLA(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False, attn_drop=0.):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
        self.attn_drop = nn.Dropout(attn_drop)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        # q[b,h,n,d],b is batch ,h is multi head, n is number of batch, d is feature
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        out = self.attn_drop(out)
        return out

class BasicBlock(nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=None):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#efficient non-local transformer block (ENLTB)
class ENLTB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0.1, attn_drop=0.1, drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio
        self.qk_scale = qk_scale
        self.conv_match1 = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)
        self.conv_match2 = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)
        self.conv_assembly = BasicBlock(default_conv, dim, dim, kernel_size, bias=qkv_bias, bn=False, act=None)

        self.norm1 = norm_layer(dim)
        self.attn = ENLA(dim_heads=dim, nb_features=dim, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        assert H == x.shape[-2] and W == x.shape[-1], "input feature has wrong size"
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        shortcut = x  # skip connection

        # Layer Norm
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        # ENLA
        x_embed_1 = self.conv_match1(x)
        x_embed_2 = self.conv_match2(x)
        x_assembly = self.conv_assembly(x)  # [B,C,H,W]
        if self.qk_scale is not None:
            x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5) * self.qk_scale
            x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5) * self.qk_scale
        else:
            x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)
            x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5)
        B, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(B, 1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(B, 1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(B, 1, H * W, -1)

        x = self.attn(x_embed_1, x_embed_2, x_assembly).squeeze(1)  # (B, H*W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)
    input_resolution = (32, 32)
    block = ENLTB(dim=64,input_resolution=input_resolution)

    print(input.size())
    output = block(input)
    print(output.size())
