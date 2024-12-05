import torch
import torch.nn as nn
import torch.nn.functional as F
#论文：Robust change detection for remote sensing images based on temporospatial interactive attention module
#论文地址：https://www.sciencedirect.com/science/article/pii/S1569843224001213

class SpatiotemporalAttentionFull(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFull, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_time_1_sf = nn.Softmax(dim=-1)
        self.energy_time_2_sf = nn.Softmax(dim=-1)
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):

        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = self.energy_time_1_sf(energy_time_1)
        energy_time_2s = self.energy_time_2_sf(energy_time_2)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)

        # energy_time_2s*g_x11*energy_space_2s = C2*S(C1) × C1*H1W1 × S(H1W1)*H2W2 = (C2*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # C2*H2W2
        # energy_time_1s*g_x12*energy_space_1s = C1*S(C2) × C2*H2W2 × S(H2W2)*H1W1 = (C1*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionBase(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionBase, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.energy_space_2s_sf = nn.Softmax(dim=-2)
        self.energy_space_1s_sf = nn.Softmax(dim=-2)

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)

        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)
        energy_space_2s = self.energy_space_2s_sf(energy_space_1)  # S(H1W1)*H2W2
        energy_space_1s = self.energy_space_1s_sf(energy_space_2)  # S(H2W2)*H1W1

        # g_x11*energy_space_2s = C1*H1W1 × S(H1W1)*H2W2 = (C1*H2W2)' is rebuild C1*H1W1
        y1 = torch.matmul(g_x11, energy_space_2s).contiguous()  # C2*H2W2
        # g_x21*energy_space_1s = C2*H2W2 × S(H2W2)*H1W1 = (C2*H1W1)' is rebuild C2*H2W2
        y2 = torch.matmul(g_x21, energy_space_1s).contiguous()
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W(y1), x2 + self.W(y2)


class SpatiotemporalAttentionFullNotWeightShared(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
        super(SpatiotemporalAttentionFullNotWeightShared, self).__init__()
        assert dimension in [2, ]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.g2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.W1 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.W2 = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        self.theta = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x1, x2):
        """
        :param x: (b, c, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x1.size(0)
        g_x11 = self.g1(x1).reshape(batch_size, self.inter_channels, -1)
        g_x12 = g_x11.permute(0, 2, 1)
        g_x21 = self.g2(x2).reshape(batch_size, self.inter_channels, -1)
        g_x22 = g_x21.permute(0, 2, 1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)

        phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
        phi_x2 = phi_x1.permute(0, 2, 1)

        energy_time_1 = torch.matmul(theta_x1, phi_x2)
        energy_time_2 = energy_time_1.permute(0, 2, 1)
        energy_space_1 = torch.matmul(theta_x2, phi_x1)
        energy_space_2 = energy_space_1.permute(0, 2, 1)

        energy_time_1s = F.softmax(energy_time_1, dim=-1)
        energy_time_2s = F.softmax(energy_time_2, dim=-1)
        energy_space_2s = F.softmax(energy_space_1, dim=-2)
        energy_space_1s = F.softmax(energy_space_2, dim=-2)
        #  C1*S(C2) energy_time_1s * C1*H1W1 g_x12 * energy_space_1s S(H2W2)*H1W1 -> C1*H1W1
        y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous()  # C2*H2W2
        #  C2*S(C1) energy_time_2s * C2*H2W2 g_x21 * energy_space_2s S(H1W1)*H2W2 -> C2*H2W2
        y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()  # C1*H1W1
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        return x1 + self.W1(y1), x2 + self.W2(y2)


if __name__ == '__main__':



    input1 = torch.randn(1, 64, 32, 32) #B C H W
    input2 = torch.randn(1, 64, 32, 32) #B C H W

    sp_full = SpatiotemporalAttentionFull(in_channels=64)
    output_full_x1, output_full_x2 = sp_full(input1, input2)

    print(input1.shape, input2.shape)
    print(output_full_x1.shape, output_full_x2.shape)


    sp_base = SpatiotemporalAttentionBase(in_channels=64)
    output_base_x1, output_base_x2 = sp_base(input1, input2)

    print(input1.shape, input2.shape)
    print(output_base_x1.shape, output_base_x2.shape)


    sp_full_not_shared = SpatiotemporalAttentionFullNotWeightShared(in_channels=64)
    output_full_not_shared_x1, output_full_not_shared_x2 = sp_full_not_shared(input1, input2)

    print(input1.shape, input2.shape)
    print(output_full_not_shared_x1.shape,
          output_full_not_shared_x2.shape)