import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class DPTAM(nn.Module):
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DPTAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print('DPTAM with kernel_size {}.'.format(kernel_size))

        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=3)#context Modeling
        self.softmax = nn.Softmax(dim=2)
        self.p1_conv1= nn.Conv1d(in_channels , in_channels, 1, bias=False)


        self.dptam = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        nt, c, h, w = x.size()

        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))

        x_22=out.view(-1,c,t)
        x22_c_t = self.p1_conv1(x_22)
        x22 =x_22.mean(2,keepdim=True)
        x22 = self.p1_conv1(x22)
        x22 = x22_c_t * x22
        x22= x_22+x22

        local_activation = self.dptam(x22).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation

        out = new_x.view(n_batch, c, t, h, w) #光local
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out



if __name__ == '__main__':
    n_segment = 16  

    block = DPTAM(in_channels=4, n_segment=n_segment)
    input = torch.rand(16, 4, 16, 16)
    output = block(input)
    print(input.size())
    print(output.size())

