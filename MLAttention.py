import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification
# 论文地址：https://arxiv.org/pdf/1811.01727.pdf

class MLAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 2)  # N, L, 1
        attention_scores = self.attention(inputs)  # N, L, hidden_size
        attention = F.softmax(attention_scores, dim=1)  # N, L, hidden_size
        attention_masked = attention * masks  # apply the mask
        return attention_masked

class FastMLAttention(nn.Module):
    def __init__(self, hidden_size):
        super(FastMLAttention, self).__init__()
        self.attention_dim = hidden_size  # Make sure this is same as your inputs dimension
        self.attention = nn.Linear(self.attention_dim, self.attention_dim)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks, attn_weights: nn.Module):
        masks = masks.unsqueeze(2)   # N, L, 1
        attention_scores = self.attention(inputs)  # N, L, hidden_size
        attention = F.softmax(attention_scores, dim=1)  # Softmax over L dimension
        attention = attention * masks  # Apply mask
        attention_masked = attention_scores * attention  # Apply attention
        return attention_masked

if __name__ == '__main__':

    batch_size = 8
    seq_len = 10
    hidden_size = 8
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    masks = torch.ones(batch_size, seq_len)

    ml_attention = MLAttention(hidden_size)
    outputs_ml = ml_attention(inputs, masks)
    print(outputs_ml.size())

    fast_ml_attention = FastMLAttention(hidden_size)
    outputs_fastml = fast_ml_attention(inputs, masks, None)
    print(outputs_fastml.size())
