from typing import Optional
import torch.nn as nn
import torch
# 论文地址：https://arxiv.org/pdf/2403.06258
# 论文：Poly Kernel Inception Network for Remote Sensing Detection(CVPR 2024)
# Github地址：https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# Context Anchor Attention (CAA) module
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # Add more normalization types if needed
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # Add more activation types if needed
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

# Example usage to print input and output shapes
if __name__ == "__main__":
    input = torch.randn(1, 64, 128, 128) #输入 B C H W
    block = CAA(64)
    output = block(input)
    print(input.size())
    print(output.size())
