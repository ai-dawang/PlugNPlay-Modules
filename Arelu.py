
import torch
import torch.nn as nn
import torch.nn.functional as F

# github地址：https://github.com/densechen/AReLU/blob/master/activations/arelu.py
# 论文：ARELU: ATTENTION-BASED RECTIFIED LINEAR UNIT
class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha