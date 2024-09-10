import torch
import torch.nn as nn
# 论文地址：https://arxiv.org/pdf/2303.16900
# 论文：InceptionNeXt: When Inception Meets ConvNeXt (CVPR 2024)
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class InceptionDWConv3d(nn.Module):
    """ Inception depthwise convolution for 3D data
    """

    def __init__(self, in_channels, cube_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hwd = nn.Conv3d(gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc)
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2),
                                   groups=gc)
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0),
                                   groups=gc)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0),
                                   groups=gc)
        self.split_indexes = (in_channels - 4 * gc, gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hwd, x_wd, x_hd, x_hw = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.dwconv_wd(x_wd), self.dwconv_hd(x_hd), self.dwconv_hw(x_hw)),
            dim=1,
        )


if __name__ == '__main__':
    block = InceptionDWConv3d(64) # 输入 C
    input = torch.randn(1, 64, 16, 224, 224) # 输入B C D H W
    output = block(input)
    print(input.size())
    print(output.size())
