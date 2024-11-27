import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#论文：LMFLOSS: A HYBRID LOSS FOR IMBALANCED MEDICAL IMAGE CLASSIFICATION


class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1 - p) ** self.gamma * logp

        return torch.mean(focal_loss)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance.
        You can start with a small value and gradually increase it to observe the impact on the model's performance.
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.

        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem.
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies
        the impact of the logits and can be useful when dealing with highly imbalanced datasets.
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class LMFLoss(nn.Module):
    def __init__(self, cls_num_list, weight, alpha=1, beta=1, gamma=2, max_m=0.5, s=30):
        super().__init__()
        self.focal_loss = FocalLoss(weight, gamma)
        self.ldam_loss = LDAMLoss(cls_num_list, max_m, weight, s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss