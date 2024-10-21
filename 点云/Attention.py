import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#论文：Point MixSwap: Attentional Point Cloud Mixing via Swapping Matched Structural Divisions
#论文地址：https://vllab.cs.nycu.edu.tw/images/paper/eccv_umam22.pdf
class Attention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_LIN, num_heads, n_pts=1024, ln=False):
        super(Attention, self).__init__()
        self.n_pts = n_pts
        self.dim_LIN = dim_LIN
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_LIN)
        self.fc_k = nn.Linear(dim_K, dim_LIN)
        self.fc_v = nn.Linear(dim_K, dim_LIN)
        if ln:
            self.ln0 = nn.LayerNorm(dim_LIN)
            self.ln1 = nn.LayerNorm(dim_LIN)
        self.fc_o = nn.Linear(dim_LIN, dim_LIN)

    def forward(self, Q, K, return_attn=True): # Q = [BS, 1, emb_dim = dim_Q]; K = [BS, n_pts, emb_dim= dim_K]
        Q = self.fc_q(Q) # [BS=6, n_div=3, dim_V=1024]
        K, V = self.fc_k(K), self.fc_v(K) # K = [BS=6, n_pts=1024, emb_dim = dim_V = 1024]; V_dim= K_dim
        dim_split = self.dim_LIN // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0) #[BS*n_head=6*n_head,n_div,dim_split=1024/1=1024] --> every n_div here & below can be n_pts
        K_ = torch.cat(K.split(dim_split, 2), 0) #[BS*n_head=6*n_head,n_pts,dim_split=1024/1=1024]
        V_ = torch.cat(V.split(dim_split, 2), 0) #[BS*n_head=6*n_head,n_pts,dim_split=1024/1=1024]
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)) / math.sqrt(self.dim_LIN), 2) #[BS*n_head=6*n_head,n_div,dim_split=1024/1=1024]
        temp = (Q_ + A.bmm(V_)).split(Q.size(0), 0) #tupple of n_head, @[BS=6,n_div=3,dim_split=1024]
        O = torch.cat(temp, 2) #[BS=6,n_div=3,dim_split*n_head=emb=1024]
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if self.num_heads >= 2:
            A = A.split(Q.size(0),dim=0) #tupple of n_head, @[BS=6,n_div=3,dim_split=1024]
            A = torch.stack([tensor_ for tensor_ in A], dim=0) #[n_head,BS,n_div=3,emb=1024]
            A = torch.mean(A, dim=0) #[BS,n_div=3,emb=1024]
        if return_attn:
            if A.size(-1) == self.n_pts:
                A = A.permute(0, 2, 1) #[BS, n_pts, n_div]
            return O, A
        else:
            return O


if __name__ == '__main__':
    # 定义注意力机制
    block = Attention(dim_Q=1024, dim_K=1024, dim_LIN=1024, num_heads=8, n_pts=1024, ln=True)

    # 创建模拟输入数据
    batch_size = 6
    Q = torch.randn(batch_size, 1, 1024)  # Query 张量
    K = torch.randn(batch_size, 1024, 1024)  # Key 张量

    # 执行前向传播
    output, attention_scores = block(Q, K, return_attn=True)

    print(output.size())
    print(attention_scores.size())
