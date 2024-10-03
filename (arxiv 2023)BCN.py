import torch
import torch.nn as nn

# 论文地址：https://arxiv.org/pdf/2312.00596
# 论文：BCN: Batch Channel Normalization for Image Classification
class BatchNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean.mul_(self.momentum).add_((1 - self.momentum) * mean.detach())
            self.runningvar.mul_(self.momentum).add_((1 - self.momentum) * variance.detach())
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


class BatchNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * variance + (self.momentum) * self.runningvar
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            return out

class BatchNormm2DViiT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2DViiT, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
            std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * std + (self.momentum) * self.runningvar
            out=(x - mean) / (std +  self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean) / torch.sqrt(
                (m / (m - 1))* self.runningvar + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            out = self.a_2 * out + self.b_2
            return out

class BatchNormm2DViTC(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2DViTC, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
            std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * std + (self.momentum) * self.runningvar
            out=(x - mean) / (std +  self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean) / torch.sqrt(
                (m / (m - 1))* self.runningvar + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            return out

class InstanceNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(InstanceNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))

        # running mean and variance should have the same dimension as in batchnorm
        # ie, a vector of size num_channels because while testing, when we get one
        # sample at a time, we should be able to use this.
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance, mean = torch.var(x, dim=[2, 3], unbiased=False), torch.mean(x, dim=[2, 3])
            out = (x - mean.view([-1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([-1, self.num_channels, 1, 1]) + self.epsilon)

        else:
            variance, mean = torch.var(x, dim=[2, 3], unbiased=False), torch.mean(x, dim=[2, 3])
            out = (x - mean.view([-1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([-1, self.num_channels, 1, 1]) + self.epsilon)

        if (self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out

class LayerNormViT(nn.Module):
     def __init__(self, features, eps=1e-6):
        super(LayerNormViT, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

     def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNormViTC(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNormViTC, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
        return(x - mean) / (std +  self.eps)


class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
      #  assert list(x.shape)[1] == self.num_channels
       # assert len(x.shape) == 4 # 4 because len((batchsize, numchannels, height, width)) = 4

        variance, mean = torch.var(x, dim = [1,2, 3], unbiased=False), torch.mean(x, dim = [1,2, 3])
        out = (x-mean.view([-1, 1, 1, 1]))/torch.sqrt(variance.view([-1, 1, 1, 1])+self.epsilon)

        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out



class LayerNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(LayerNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False), torch.mean(x, dim=[1, 2, 3])

        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1, 1, 1, 1]) + self.epsilon)
        return out


class GroupNorm2D(nn.Module):
    def __init__(self, num_channels, num_groups=4, epsilon=1e-5):
        super(GroupNorm2D, self).__init__()
        self.num_channels = num_channels
        # self.num_groups = num_groups
        self.num_groups = num_channels // 4
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)
        [N, C, H, W] = list(x.shape)

        out = torch.reshape(x, (N, self.num_groups, self.num_channels // self.num_groups, H, W))
        variance, mean = torch.var(out, dim=[2, 3, 4], unbiased=False, keepdim=True), torch.mean(out, dim=[2, 3, 4],
                                                                                                 keepdim=True)
        out = (out - mean) / torch.sqrt(variance + self.epsilon)
        out = out.view(N, self.num_channels, H, W)
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


class BatchNorm_ByoL(nn.Module):
    def __init__(self, bn, num_channels=2048, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNorm_ByoL, self).__init__()
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = epsilon
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        std = self.runningvar.add(self.eps).sqrt()
        return x.sub(self.runningmean).div(std).mul(self.gamma).add(self.beta)


class LaychNorm_ByoL(nn.Module):
    def __init__(self, bn, num_channels=2048, epsilon=1e-5, momentum=0.9, rescale=True):
        super(LaychNorm_ByoL, self).__init__()
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = epsilon
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        std = self.runningvar.add(self.eps).sqrt()
        return x.sub(self.runningmean).div(std).mul(self.gamma).add(self.beta)


class BatchNorm_Byol(nn.Module):
    def __init__(self, bn, num_channels=2048, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNorm_Byol, self).__init__()
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = epsilon
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        std = self.runningvar.add(self.eps).sqrt()
        return x.sub(self.runningmean).div(std)


class LaychNorm_Byol(nn.Module):
    def __init__(self, bn, num_channels=2048, epsilon=1e-5, momentum=0.9, rescale=True):
        super(LaychNorm_Byol, self).__init__()
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = epsilon
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        std = self.runningvar.add(self.eps).sqrt()
        return x.sub(self.runningmean).div(std)


class BatchChannelNorm_Byol(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm_Byol, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.Batchh = BatchNorm_Byol(self.num_channels, epsilon=self.epsilon)
        self.layeer = LaychNorm_Byol(self.num_channels, epsilon=self.epsilon)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var * X + 1 - self.BCN_var * Y
        out = self.gamma * out + self.beta
        return out


class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.Batchh = BatchNormm2D(self.num_channels, epsilon=self.epsilon)
        self.layeer = LayerNormm2D(self.num_channels, epsilon=self.epsilon)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out

class BatchChannelNormvit(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
            super(BatchChannelNormvit, self).__init__()
            self.num_channels = num_channels
            self.epsilon = epsilon
            self.momentum = momentum
            self.Batchh = BatchNormm2DViTC(self.num_channels, epsilon=self.epsilon)
            self.layeer = LayerNormViTC(self.num_channels)
            # The BCN variable to be learnt
            self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
            # Gamma and Beta for rescaling
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
            X = self.Batchh(x)
            Y = self.layeer(x)
            out = self.BCN_var * X + (
                    1 - self.BCN_var) * Y
            out = self.gamma* out + self.beta
            return out



if __name__ == '__main__':
    block = BatchChannelNorm(num_channels=64)
    input = torch.rand(64, 64, 9, 9)
    output = block(input)
    print(input.size())
    print(output.size())