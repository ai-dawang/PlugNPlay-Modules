import numpy as np
from typing import Union, Sequence, Tuple, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Any, Callable
from torchvision.ops import StochasticDepth as StochasticDepthTorch
# 论文地址：https://arxiv.org/pdf/2407.07720v1
# 论文：SvANet: A Scale-variant Attention-based Network for Small Medical Object Segmentation
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
class Dropout(nn.Dropout):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super(Dropout, self).__init__(p=p, inplace=inplace)

class StochasticDepth(StochasticDepthTorch):
    def __init__(self, p: float, Mode: str="row") -> None:
        super().__init__(p, Mode)

def pair(Val):
    return Val if isinstance(Val, (tuple, list)) else (Val, Val)

def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.Py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        DimEmbed (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        AttnDropRate (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        DimEmbed: int,
        AttnDropRate: Optional[float]=0.0,
        Bias: Optional[bool]=True,
    ) -> None:
        super().__init__()

        self.qkv_proj = BaseConv2d(DimEmbed, 1 + (2 * DimEmbed), 1, bias=Bias)

        self.AttnDropRate = Dropout(p=AttnDropRate)
        self.out_proj = BaseConv2d(DimEmbed, DimEmbed, 1, bias=Bias)
        self.DimEmbed = DimEmbed

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.DimEmbed, self.DimEmbed], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.AttnDropRate(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(
            self,
            DimEmbed: int,
            DimFfnLatent: int,
            AttnDropRate: Optional[float] = 0.0,
            DropRate: Optional[float] = 0.1,
            FfnDropRate: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        AttnUnit = LinearSelfAttention(DimEmbed, AttnDropRate, Bias=True)

        self.PreNormAttn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            AttnUnit,
            Dropout(DropRate),
        )

        self.PreNormFfn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            BaseConv2d(DimEmbed, DimFfnLatent, 1, 1, ActLayer=nn.SiLU),
            Dropout(FfnDropRate),
            BaseConv2d(DimFfnLatent, DimEmbed, 1, 1),
            Dropout(DropRate),
        )

        self.DimEmbed = DimEmbed

    def forward(self, x: Tensor) -> Tensor:
        # self-attention
        x = x + self.PreNormAttn(x)

        # Feed forward network
        x = x + self.PreNormFfn(x)
        return x

class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            # norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

class BaseFormer(nn.Module):
    def __init__(
            self,
            InChannels: int,
            FfnMultiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            NumAttnBlocks: Optional[int] = 2,
            AttnDropRate: Optional[float] = 0.0,
            DropRate: Optional[float] = 0.0,
            FfnDropRate: Optional[float] = 0.0,
            PatchRes: Optional[int] = 2,
            Dilation: Optional[int] = 1,
            ViTSELayer: Optional[nn.Module] = None,
            **kwargs: Any,
    ) -> None:
        DimAttnUnit = InChannels // 2
        DimCNNOut = DimAttnUnit

        Conv3x3In = BaseConv2d(
            InChannels, InChannels, 3, 1, dilation=Dilation,
            BNorm=True, ActLayer=nn.SiLU,
        )  # depth-wise separable convolution
        ViTSELayer = ViTSELayer(InChannels, **kwargs) if ViTSELayer is not None else nn.Identity()
        Conv1x1In = BaseConv2d(InChannels, DimCNNOut, 1, 1, bias=False)

        super(BaseFormer, self).__init__()
        self.LocalRep = nn.Sequential(Conv3x3In, ViTSELayer, Conv1x1In)

        self.GlobalRep, DimAttnUnit = self.buildAttnLayer(
            DimAttnUnit, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate, FfnDropRate,
        )
        self.ConvProj = BaseConv2d(DimCNNOut, InChannels, 1, 1, BNorm=True)

        self.DimCNNOut = DimCNNOut

        self.HPatch, self.WPatch = pair(PatchRes)
        self.PatchArea = self.WPatch * self.HPatch

    def buildAttnLayer(
            self,
            DimModel: int,
            FfnMult: Union[Sequence, int, float],
            NumAttnBlocks: int,
            AttnDropRate: float,
            DropRate: float,
            FfnDropRate: float,
    ) -> Tuple[nn.Module, int]:

        if isinstance(FfnMult, Sequence) and len(FfnMult) == 2:
            DimFfn = (
                    np.linspace(FfnMult[0], FfnMult[1], NumAttnBlocks, dtype=float) * DimModel
            )
        elif isinstance(FfnMult, Sequence) and len(FfnMult) == 1:
            DimFfn = [FfnMult[0] * DimModel] * NumAttnBlocks
        elif isinstance(FfnMult, (int, float)):
            DimFfn = [FfnMult * DimModel] * NumAttnBlocks
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        DimFfn = [makeDivisible(d, 16) for d in DimFfn]

        GlobalRep = [
            LinearAttnFFN(DimModel, DimFfn[block_idx], AttnDropRate, DropRate, FfnDropRate)
            for block_idx in range(NumAttnBlocks)
        ]
        GlobalRep.append(nn.BatchNorm2d(DimModel))
        return nn.Sequential(*GlobalRep), DimModel

    def unfolding(self, FeatureMap: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        B, C, H, W = FeatureMap.shape

        # [B, C, H, W] --> [B, C, P, N]
        Patches = F.unfold(
            FeatureMap,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )
        Patches = Patches.reshape(
            B, C, self.HPatch * self.WPatch, -1
        )

        return Patches, (H, W)

    def folding(self, Patches: Tensor, OutputSize: Tuple[int, int]) -> Tensor:
        B, C, P, N = Patches.shape  # BatchSize, DimIn, PatchSize, NumPatches

        # [B, C, P, N]
        Patches = Patches.reshape(B, C * P, N)

        FeatureMap = F.fold(
            Patches,
            output_size=OutputSize,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )

        return FeatureMap

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        Fm = self.LocalRep(x)

        # convert feature map to patches
        Patches, OutputSize = self.unfolding(Fm)

        # learn global representations on all patches
        Patches = self.GlobalRep(Patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        Fm = self.folding(Patches, OutputSize)
        Fm = self.ConvProj(Fm)

        return Fm

#AssemFormer, a method that combines convolution with a vision transformer by assembling tensors.
class AssemFormer(BaseFormer):
    """
    Inspired by MobileViTv3.
    Adapted from https://github.com/micronDLA/MobileViTv3/blob/main/MobileViTv3-v2/cvnets/modules/mobilevit_block.py
    """

    def __init__(
            self,
            InChannels: int,
            FfnMultiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
            NumAttnBlocks: Optional[int] = 2,
            AttnDropRate: Optional[float] = 0.0,
            DropRate: Optional[float] = 0.0,
            FfnDropRate: Optional[float] = 0.0,
            PatchRes: Optional[int] = 2,
            Dilation: Optional[int] = 1,
            SDProb: Optional[float] = 0.0,
            ViTSELayer: Optional[nn.Module] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(InChannels, FfnMultiplier, NumAttnBlocks, AttnDropRate,
                         DropRate, FfnDropRate, PatchRes, Dilation, ViTSELayer, **kwargs)
        # AssembleFormer: input changed from just global to local + global
        self.ConvProj = BaseConv2d(2 * self.DimCNNOut, InChannels, 1, 1, BNorm=True)

        self.Dropout = StochasticDepth(SDProb)

    def forward(self, x: Tensor) -> Tensor:
        FmConv = self.LocalRep(x)

        # convert feature map to patches
        Patches, OutputSize = self.unfolding(FmConv)

        # learn global representations on all patches
        Patches = self.GlobalRep(Patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        Fm = self.folding(Patches, OutputSize)

        # AssembleFormer: local + global instead of only global
        Fm = self.ConvProj(torch.cat((Fm, FmConv), dim=1))

        # AssembleFormer: skip connection
        return x + self.Dropout(Fm)


if __name__ == '__main__':
    input = torch.randn(1, 64, 128, 128)# 输入 B C H W
    block = AssemFormer(InChannels=64)
    output = block(input)
    print(output.size())