# 论文：Frequency-Adaptive Dilated Convolution for Semantic Segmentation[CVPR 2024]
# 论文地址：https://arxiv.org/abs/2403.05369
import torch
import torch.nn as nn
import torch.fft


class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


import torch.nn.functional as F


def generate_laplacian_pyramid(input_tensor, num_levels, size_align=True, mode='bilinear'):
    """"
    a alternative way for feature frequency decompose
    """
    pyramid = []
    current_tensor = input_tensor
    _, _, H, W = current_tensor.shape
    for _ in range(num_levels):
        b, _, h, w = current_tensor.shape
        downsampled_tensor = F.interpolate(current_tensor, (h // 2 + h % 2, w // 2 + w % 2), mode=mode,
                                           align_corners=(H % 2) == 1)  # antialias=True
        if size_align:
            # upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode='bilinear', align_corners=(H%2) == 1)
            # laplacian = current_tensor - upsampled_tensor
            # laplacian = F.interpolate(laplacian, (H, W), mode='bilinear', align_corners=(H%2) == 1)
            upsampled_tensor = F.interpolate(downsampled_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1)
            laplacian = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1) - upsampled_tensor
            # print(laplacian.shape)
        else:
            upsampled_tensor = F.interpolate(downsampled_tensor, (h, w), mode=mode, align_corners=(H % 2) == 1)
            laplacian = current_tensor - upsampled_tensor
        pyramid.append(laplacian)
        current_tensor = downsampled_tensor
    if size_align: current_tensor = F.interpolate(current_tensor, (H, W), mode=mode, align_corners=(H % 2) == 1)
    pyramid.append(current_tensor)
    return pyramid


class FrequencySelection(nn.Module):
    def __init__(self,
                 in_channels,
                 k_list=[2],
                 # freq_list=[2, 3, 5, 7, 9, 11],
                 lowfreq_att=True,
                 fs_feat='feat',
                 lp_type='freq',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 global_selection=False,
                 ):
        super().__init__()
        # k_list.sort()
        # print()
        self.k_list = k_list
        # self.freq_list = freq_list
        self.lp_list = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat = fs_feat
        self.lp_type = lp_type
        self.in_channels = in_channels
        # self.residual = residual
        if spatial_group > 64: spatial_group = in_channels
        self.spatial_group = spatial_group
        self.lowfreq_att = lowfreq_att
        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:  _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(in_channels=in_channels,
                                             out_channels=self.spatial_group,
                                             stride=1,
                                             kernel_size=spatial_kernel,
                                             groups=self.spatial_group,
                                             padding=spatial_kernel // 2,
                                             bias=True)
                if init == 'zero':
                    freq_weight_conv.weight.data.zero_()
                    freq_weight_conv.bias.data.zero_()
                else:
                    # raise NotImplementedError
                    pass
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError

        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad2d(padding=k // 2),
                    # nn.ZeroPad2d(padding= k // 2),
                    nn.AvgPool2d(kernel_size=k, padding=0, stride=1)
                ))
        elif self.lp_type == 'laplacian':
            pass
        elif self.lp_type == 'freq':
            pass
        else:
            raise NotImplementedError

        self.act = act
        # self.freq_weight_conv_list.append(nn.Conv2d(self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1], 1, kernel_size=1, padding=0, bias=True))
        self.global_selection = global_selection
        if self.global_selection:
            self.global_selection_conv_real = nn.Conv2d(in_channels=in_channels,
                                                        out_channels=self.spatial_group,
                                                        stride=1,
                                                        kernel_size=1,
                                                        groups=self.spatial_group,
                                                        padding=0,
                                                        bias=True)
            self.global_selection_conv_imag = nn.Conv2d(in_channels=in_channels,
                                                        out_channels=self.spatial_group,
                                                        stride=1,
                                                        kernel_size=1,
                                                        groups=self.spatial_group,
                                                        padding=0,
                                                        bias=True)
            if init == 'zero':
                self.global_selection_conv_real.weight.data.zero_()
                self.global_selection_conv_real.bias.data.zero_()
                self.global_selection_conv_imag.weight.data.zero_()
                self.global_selection_conv_imag.bias.data.zero_()

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            freq_weight = freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            freq_weight = freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError
        return freq_weight

    def forward(self, x, att_feat=None):
        """
        att_feat:feat for gen att
        """
        # freq_weight = self.freq_weight_conv(x)
        # self.sp_act(freq_weight)
        # if self.residual: x_residual = x.clone()
        if att_feat is None: att_feat = x
        x_list = []
        if self.lp_type == 'avgpool':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            pre_x = x
            b, _, h, w = x.shape
            for idx, avg in enumerate(self.lp_list):
                low_part = avg(x)
                high_part = pre_x - low_part
                pre_x = low_part
                # x_list.append(freq_weight[:, idx:idx+1] * high_part)
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
                                                                                               -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre_x)
        elif self.lp_type == 'laplacian':
            # for avg, freq_weight in zip(self.avg_list, self.freq_weight_conv_list):
            # pre_x = x
            b, _, h, w = x.shape
            pyramids = generate_laplacian_pyramid(x, len(self.k_list), size_align=True)
            # print('pyramids', len(pyramids))
            for idx, avg in enumerate(self.k_list):
                # print(idx)
                high_part = pyramids[idx]
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
                                                                                               -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pyramids[-1].reshape(b, self.spatial_group,
                                                                                                  -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pyramids[-1])
        elif self.lp_type == 'freq':
            pre_x = x.clone()
            b, _, h, w = x.shape
            # b, _c, h, w = freq_weight.shape
            # freq_weight = freq_weight.reshape(b, self.spatial_group, -1, h, w)
            x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))
            if self.global_selection:
                # global_att_real = self.global_selection_conv_real(x_fft.real)
                # global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # global_att_imag = self.global_selection_conv_imag(x_fft.imag)
                # global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # x_fft = x_fft.reshape(b, self.spatial_group, -1, h, w)
                # x_fft.real *= global_att_real
                # x_fft.imag *= global_att_imag
                # x_fft = x_fft.reshape(b, -1, h, w)
                # 将x_fft复数拆分成实部和虚部
                x_real = x_fft.real
                x_imag = x_fft.imag
                # 计算实部的全局注意力
                global_att_real = self.global_selection_conv_real(x_real)
                global_att_real = self.sp_act(global_att_real).reshape(b, self.spatial_group, -1, h, w)
                # 计算虚部的全局注意力
                global_att_imag = self.global_selection_conv_imag(x_imag)
                global_att_imag = self.sp_act(global_att_imag).reshape(b, self.spatial_group, -1, h, w)
                # 重塑x_fft为形状为(b, self.spatial_group, -1, h, w)的张量
                x_real = x_real.reshape(b, self.spatial_group, -1, h, w)
                x_imag = x_imag.reshape(b, self.spatial_group, -1, h, w)
                # 分别应用实部和虚部的全局注意力
                x_fft_real_updated = x_real * global_att_real
                x_fft_imag_updated = x_imag * global_att_imag
                # 合并为复数
                x_fft_updated = torch.complex(x_fft_real_updated, x_fft_imag_updated)
                # 重塑x_fft为形状为(b, -1, h, w)的张量
                x_fft = x_fft_updated.reshape(b, -1, h, w)

            for idx, freq in enumerate(self.k_list):
                mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
                mask[:, :, round(h / 2 - h / (2 * freq)):round(h / 2 + h / (2 * freq)),
                round(w / 2 - w / (2 * freq)):round(w / 2 + w / (2 * freq))] = 1.0
                low_part = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
                high_part = pre_x - low_part
                pre_x = low_part
                freq_weight = self.freq_weight_conv_list[idx](att_feat)
                freq_weight = self.sp_act(freq_weight)
                # tmp = freq_weight[:, :, idx:idx+1] * high_part.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group,
                                                                                               -1, h, w)
                x_list.append(tmp.reshape(b, -1, h, w))
            if self.lowfreq_att:
                freq_weight = self.freq_weight_conv_list[len(x_list)](att_feat)
                # tmp = freq_weight[:, :, len(x_list):len(x_list)+1] * pre_x.reshape(b, self.spatial_group, -1, h, w)
                tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h,
                                                                                           w)
                x_list.append(tmp.reshape(b, -1, h, w))
            else:
                x_list.append(pre_x)
        x = sum(x_list)
        return x


from mmcv.ops.deform_conv import DeformConv2dPack
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack, \
    CONV_LAYERS
import torch_dct as dct
#pip install torch-dct


class AdaptiveDilatedConv(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args,
                 offset_freq=None,  # deprecated
                 padding_mode='repeat',
                 kernel_decompose='both',
                 conv_type='conv',
                 sp_att=False,
                 pre_fs=True,  # False, use dilation
                 epsilon=1e-4,
                 use_zero_dilation=False,
                 use_dct=False,
                 fs_cfg={
                     'k_list': [2, 4, 8],
                     'fs_feat': 'feat',
                     'lowfreq_att': False,
                     'lp_type': 'freq',
                     # 'lp_type':'laplacian',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                 },
                 **kwargs):
        super().__init__(*args, **kwargs)
        if padding_mode == 'zero':
            self.PAD = nn.ZeroPad2d(self.kernel_size[0] // 2)
        elif padding_mode == 'repeat':
            self.PAD = nn.ReplicationPad2d(self.kernel_size[0] // 2)
        else:
            self.PAD = nn.Identity()

        self.kernel_decompose = kernel_decompose
        self.use_dct = use_dct

        if kernel_decompose == 'both':
            self.OMNI_ATT1 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                           groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
            self.OMNI_ATT2 = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels,
                                           kernel_size=self.kernel_size[0] if self.use_dct else 1, groups=1,
                                           reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'high':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                          groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        elif kernel_decompose == 'low':
            self.OMNI_ATT = OmniAttention(in_planes=self.in_channels, out_planes=self.out_channels, kernel_size=1,
                                          groups=1, reduction=0.0625, kernel_num=1, min_channel=16)
        self.conv_type = conv_type
        if conv_type == 'conv':
            self.conv_offset = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
                dilation=1,
                bias=True)

        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 1 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
            dilation=1,
            bias=True)
        if sp_att:
            self.conv_mask_mean_level = nn.Conv2d(
                self.in_channels,
                self.deform_groups * 1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.kernel_size[0] // 2 if isinstance(self.PAD, nn.Identity) else 0,
                dilation=1,
                bias=True)

        self.offset_freq = offset_freq



        # An offset is like [y0, x0, y1, x1, y2, x2, ⋯, y8, x8]
        offset = [-1, -1, -1, 0, -1, 1,
                  0, -1, 0, 0, 0, 1,
                  1, -1, 1, 0, 1, 1]
        offset = torch.Tensor(offset)
        # offset[0::2] *= self.dilation[0]
        # offset[1::2] *= self.dilation[1]
        # a tuple of two ints – in which case, the first int is used for the height dimension, and the second int for the width dimension
        self.register_buffer('dilated_offset', torch.Tensor(offset[None, None, ..., None, None]))  # B, G, 18, 1, 1
        if fs_cfg is not None:
            if pre_fs:
                self.FS = FrequencySelection(self.in_channels, **fs_cfg)
            else:
                self.FS = FrequencySelection(1, **fs_cfg)  # use dilation
        self.pre_fs = pre_fs
        self.epsilon = epsilon
        self.use_zero_dilation = use_zero_dilation
        self.init_weights()

    def freq_select(self, x):
        if self.offset_freq is None:
            res = x
        elif self.offset_freq in ('FLC_high', 'SLP_high'):
            res = x - self.LP(x)
        elif self.offset_freq in ('FLC_res', 'SLP_res'):
            res = 2 * x - self.LP(x)
        else:
            raise NotImplementedError
        return res

    def init_weights(self):
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            # if isinstanace(self.conv_offset, nn.Conv2d):
            if self.conv_type == 'conv':
                self.conv_offset.weight.data.zero_()
                # self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + 1e-4)
                self.conv_offset.bias.data.fill_((self.dilation[0] - 1) / self.dilation[0] + self.epsilon)
            # self.conv_offset.bias.data.zero_()
        # if hasattr(self, 'conv_offset'):
        # self.conv_offset_low[1].weight.data.zero_()
        # if hasattr(self, 'conv_offset_high'):
        # self.conv_offset_high[1].weight.data.zero_()
        # self.conv_offset_high[1].bias.data.zero_()
        if hasattr(self, 'conv_mask'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

        if hasattr(self, 'conv_mask_mean_level'):
            self.conv_mask.weight.data.zero_()
            self.conv_mask.bias.data.zero_()

    # @force_fp32(apply_to=('x',))
    # @force_fp32
    def forward(self, x):
        # offset = self.conv_offset(self.freq_select(x)) + self.conv_offset_low(self.freq_select(x))
        if hasattr(self, 'FS') and self.pre_fs: x = self.FS(x)
        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            c_att1, f_att1, _, _, = self.OMNI_ATT1(x)
            c_att2, f_att2, spatial_att2, _, = self.OMNI_ATT2(x)
        elif hasattr(self, 'OMNI_ATT'):
            c_att, f_att, _, _, = self.OMNI_ATT(x)

        if self.conv_type == 'conv':
            offset = self.conv_offset(self.PAD(self.freq_select(x)))
        elif self.conv_type == 'multifreqband':
            offset = self.conv_offset(self.freq_select(x))
        # high_gate = self.conv_offset_high(x)
        # high_gate = torch.exp(-0.5 * high_gate ** 2)
        # offset = F.relu(offset, inplace=True) * self.dilation[0] - 1 # ensure > 0
        if self.use_zero_dilation:
            offset = (F.relu(offset + 1, inplace=True) - 1) * self.dilation[0]  # ensure > 0
        else:
            # offset = F.relu(offset, inplace=True) * self.dilation[0] # ensure > 0
            offset = offset.abs() * self.dilation[0]  # ensure > 0
            # offset[offset<0] = offset[offset<0].exp() - 1
        # print(offset.mean(), offset.std(), offset.max(), offset.min())
        if hasattr(self, 'FS') and (self.pre_fs == False): x = self.FS(x, F.interpolate(offset, x.shape[-2:],
                                                                                        mode='bilinear', align_corners=(
                                                                                                                                   x.shape[
                                                                                                                                       -1] % 2) == 1))
        # print(offset.max(), offset.abs().min(), offset.abs().mean())
        # offset *= high_gate # ensure > 0
        b, _, h, w = offset.shape
        offset = offset.reshape(b, self.deform_groups, -1, h, w) * self.dilated_offset
        # offset = offset.reshape(b, self.deform_groups, -1, h, w).repeat(1, 1, 9, 1, 1)
        # offset[:, :, 0::2, ] *= self.dilated_offset[:, :, 0::2, ]
        # offset[:, :, 1::2, ] *= self.dilated_offset[:, :, 1::2, ]
        offset = offset.reshape(b, -1, h, w)

        x = self.PAD(x)
        mask = self.conv_mask(x)
        mask = mask.sigmoid()
        # print(mask.shape)
        # mask = mask.reshape(b, self.deform_groups, -1, h, w).softmax(dim=2)
        if hasattr(self, 'conv_mask_mean_level'):
            mask_mean_level = torch.sigmoid(self.conv_mask_mean_level(x)).reshape(b, self.deform_groups, -1, h, w)
            mask = mask * mask_mean_level
        mask = mask.reshape(b, -1, h, w)

        if hasattr(self, 'OMNI_ATT1') and hasattr(self, 'OMNI_ATT2'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            adaptive_weight_res = adaptive_weight - adaptive_weight_mean
            _, c_out, c_in, k, k = adaptive_weight.shape
            if self.use_dct:
                dct_coefficients = dct.dct_2d(adaptive_weight_res)
                # print(adaptive_weight_res.shape, dct_coefficients.shape)
                spatial_att2 = spatial_att2.reshape(b, 1, 1, k, k)
                dct_coefficients = dct_coefficients * (spatial_att2 * 2)
                # print(dct_coefficients.shape)
                adaptive_weight_res = dct.idct_2d(dct_coefficients)
                # adaptive_weight_res = adaptive_weight_res.reshape(b, c_out, c_in, k, k)
                # print(adaptive_weight_res.shape, dct_coefficients.shape)
            # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
            # adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (f_att1.unsqueeze(2) * 2) + (adaptive_weight - adaptive_weight_mean) * (c_att2.unsqueeze(1) * 2) * (f_att2.unsqueeze(2) * 2)
            adaptive_weight = adaptive_weight_mean * (c_att1.unsqueeze(1) * 2) * (
                        f_att1.unsqueeze(2) * 2) + adaptive_weight_res * (c_att2.unsqueeze(1) * 2) * (
                                          f_att2.unsqueeze(2) * 2)
            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            if self.bias is not None:
                bias = self.bias.repeat(b)
            else:
                bias = self.bias
            # print(adaptive_weight.shape)
            # print(bias.shape)
            # print(x.shape)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, bias,
                                        self.stride,
                                        (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
                                                                                                           nn.Identity) else (
                                        0, 0),  # padding
                                        (1, 1),  # dilation
                                        self.groups * b, self.deform_groups * b)
        elif hasattr(self, 'OMNI_ATT'):
            offset = offset.reshape(1, -1, h, w)
            mask = mask.reshape(1, -1, h, w)
            x = x.reshape(1, -1, x.size(-2), x.size(-1))
            adaptive_weight = self.weight.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # b, c_out, c_in, k, k
            adaptive_weight_mean = adaptive_weight.mean(dim=(-1, -2), keepdim=True)
            # adaptive_weight = adaptive_weight_mean * (2 * c_att.unsqueeze(1)) * (2 * f_att.unsqueeze(2)) + adaptive_weight - adaptive_weight_mean
            if self.kernel_decompose == 'high':
                adaptive_weight = adaptive_weight_mean + (adaptive_weight - adaptive_weight_mean) * (
                            c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2)
            elif self.kernel_decompose == 'low':
                adaptive_weight = adaptive_weight_mean * (c_att.unsqueeze(1) * 2) * (f_att.unsqueeze(2) * 2) + (
                            adaptive_weight - adaptive_weight_mean)

            adaptive_weight = adaptive_weight.reshape(-1, self.in_channels // self.groups, 3, 3)
            # adaptive_bias = self.unsqueeze(0).repeat(b, 1, 1, 1, 1)
            # print(adaptive_weight.shape)
            # print(offset.shape)
            # print(mask.shape)
            # print(x.shape)
            x = modulated_deform_conv2d(x, offset, mask, adaptive_weight, self.bias,
                                        self.stride,
                                        (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
                                                                                                           nn.Identity) else (
                                        0, 0),  # padding
                                        (1, 1),  # dilation
                                        self.groups * b, self.deform_groups * b)
        else:
            x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                        self.stride,
                                        (self.kernel_size[0] // 2, self.kernel_size[1] // 2) if isinstance(self.PAD,
                                                                                                           nn.Identity) else (
                                        0, 0),  # padding
                                        (1, 1),  # dilation
                                        self.groups, self.deform_groups)
        # x = modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
        #                                self.stride, self.padding,
        #                                self.dilation, self.groups,
        #                                self.deform_groups)
        # if hasattr(self, 'OMNI_ATT'): x = x * f_att
        return x.reshape(b, -1, h, w)


if __name__ == '__main__':

    input_tensor = torch.randn(2, 64, 128, 128)

    adaptive_dilated_conv = AdaptiveDilatedConv(in_channels=64,out_channels=64,kernel_size=3)

    output_tensor = adaptive_dilated_conv(input_tensor)

    print(input_tensor.shape)
    print(output_tensor.shape)