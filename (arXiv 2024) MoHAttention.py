import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.layers import use_fused_attn

# 论文: Multi-Head Attention as Mixture-of-Head Attention

# 论文地址：https://arxiv.org/pdf/2410.11842


class MoHAttention(nn.Module):
    fused_attn: Final[bool]
    LOAD_BALANCING_LOSSES = []

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            shared_head=0,
            routed_head=0,
            head_dim=None,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        
        if head_dim is None:
            self.head_dim = dim // num_heads
        else:
            self.head_dim = head_dim
        
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, (self.head_dim * self.num_heads) * 3, bias=qkv_bias)
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        
        self.proj_drop = nn.Dropout(proj_drop)

        self.shared_head = shared_head
        self.routed_head = routed_head
        
        if self.routed_head > 0:
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        _x = x.reshape(B * N, C)
        
        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)

            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)

            if self.training:
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)
                l_aux = torch.mean(me * ce) * num_experts * num_experts

                MoHAttention.LOAD_BALANCING_LOSSES.append(l_aux)

            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        if self.routed_head > 0:
            x = x.transpose(1, 2)

            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head

                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
        
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:,:,1], routed_head_gates)
                
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)

            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def main():

    batch_size = 2
    num_tokens = 16
    embed_dim = 64

    input = torch.rand(batch_size, num_tokens, embed_dim)

    num_heads = 4
    attn_layer = MoHAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.1,
        proj_drop=0.1,
        shared_head=2,
        routed_head=2,
        head_dim=16
    )


    attn_layer.train()

    output = attn_layer(input)

    print(input.size())
    print(output.size())

if __name__ == "__main__":
    main()
