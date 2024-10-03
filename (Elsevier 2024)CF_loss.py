import torch.nn as nn
import torch
import torch.nn.functional as F
# 论文：CF-Loss: Clinically-relevant feature optimised loss function for retinal multi-class vessel segmentation and vascular feature measurement
def encode_mask_3d(ground_truth, num_classes=4):
    batch_size, _, depth, height, width = ground_truth.size()
    one_hot = torch.zeros((batch_size, num_classes, depth, height, width), device=ground_truth.device)
    ground_truth = ground_truth.long()
    one_hot = one_hot.scatter_(1, ground_truth, 1)
    return one_hot

class CF_Loss_3D(nn.Module):
    def __init__(self, img_depth, beta, alpha, gamma):
        super(CF_Loss_3D, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.img_depth = img_depth
        self.CE = nn.CrossEntropyLoss()
        self.p = torch.tensor([img_depth], dtype=torch.float, device='cuda')
        self.n = torch.log(self.p) / torch.log(torch.tensor([2.0], device='cuda'))
        self.n = torch.floor(self.n)
        self.sizes = 2 ** torch.arange(self.n.item(), 1, -1, device='cuda').to(dtype=torch.int)

    def get_count_3d(self, sizes, p, masks_pred_softmax):
        counts = torch.zeros((masks_pred_softmax.shape[0], len(sizes), 2), device='cuda')
        index = 0

        for size in sizes:
            # 对3D数据使用3D池化
            stride = (1, size, size)  # 保持深度方向不变
            pool = nn.AvgPool3d(kernel_size=(1, size, size), stride=stride)

            S = pool(masks_pred_softmax)
            S = S * ((S > 0) & (S < (size * size)))
            counts[..., index, 0] = (S[:, 0, ...] - S[:, 2, ...]).abs().sum() / (S[:, 2, ...] > 0).sum()
            counts[..., index, 1] = (S[:, 1, ...] - S[:, 3, ...]).abs().sum() / (S[:, 3, ...] > 0).sum()

            index += 1

        return counts

    def forward(self, prediction, ground_truth):
        # 假设ground_truth已经是适当格式
        ground_truth_encoded = encode_mask_3d(ground_truth)  # 需要定义适用于3D数据的encode_mask_3d
        prediction_softmax = F.softmax(prediction, dim=1)

        loss_CE = self.CE(prediction_softmax, ground_truth.squeeze(1).long())

        Loss_vd = (torch.abs(prediction_softmax[:, 1, ...].sum() - ground_truth_encoded[:, 1, ...].sum()) + torch.abs(prediction_softmax[:, 2, ...].sum() - ground_truth_encoded[:, 2, ...].sum())) / (prediction_softmax.shape[0] * prediction_softmax.shape[2] * prediction_softmax.shape[3] * prediction_softmax.shape[4])

        prediction_softmax = prediction_softmax[:, 1:3, ...]
        ground_truth_encoded = ground_truth_encoded[:, 1:3, ...]
        combined = torch.cat((prediction_softmax, ground_truth_encoded), 1)
        counts = self.get_count_3d(self.sizes, self.p, combined)

        artery_ = torch.sqrt(torch.sum(self.sizes * ((counts[..., 0]) ** 2)))
        vein_ = torch.sqrt(torch.sum(self.sizes * ((counts[..., 1]) ** 2)))
        size_t = torch.sqrt(torch.sum(self.sizes ** 2))
        loss_FD = (artery_ + vein_) / size_t / prediction_softmax.shape[0]

        loss_value = self.beta * loss_CE + self.alpha * loss_FD + self.gamma * Loss_vd

        return loss_value


