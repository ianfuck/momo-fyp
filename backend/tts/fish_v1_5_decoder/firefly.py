import math
from functools import partial
from math import prod
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.checkpoint import checkpoint

from fish_speech.models.dac.modded_dac import DAC


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv1D") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]):
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    if isinstance(n_frames, torch.Tensor):
        ideal_length = (torch.ceil(n_frames).long() - 1) * stride + (
            kernel_size - padding_total
        )
    else:
        ideal_length = (math.ceil(n_frames) - 1) * stride + (
            kernel_size - padding_total
        )
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


class FishConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation

    def forward(self, x):
        pad = self.kernel_size - self.stride
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self


class FishTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_parametrizations(self, name="weight"):
        self.conv = remove_parametrizations(self.conv, name)
        return self


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[0]
                ).weight_norm(),
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[1]
                ).weight_norm(),
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[2]
                ).weight_norm(),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[0]
                ).weight_norm(),
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[1]
                ).weight_norm(),
                FishConvNet(
                    channels, channels, kernel_size, stride=1, dilation=dilation[2]
                ).weight_norm(),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for conv in self.convs1:
            conv.remove_parametrizations()
        for conv in self.convs2:
            conv.remove_parametrizations()


class ParallelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilation_sizes)

        self.blocks = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock1(channels, kernel_size, dilation))

    def forward(self, x):
        return torch.stack([block(x) for block in self.blocks], dim=0).mean(dim=0)

    def remove_parametrizations(self):
        for block in self.blocks:
            block.remove_parametrizations()


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
    ):
        super().__init__()

        assert prod(upsample_rates) == hop_length, f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = FishConvNet(
            num_mels,
            upsample_initial_channel,
            pre_conv_kernel_size,
            stride=1,
        ).weight_norm()

        self.num_upsamples = len(upsample_rates)

        self.ups = nn.ModuleList()
        for index, (upsample_rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                FishTransConvNet(
                    upsample_initial_channel // (2**index),
                    upsample_initial_channel // (2 ** (index + 1)),
                    kernel_size,
                    stride=upsample_rate,
                ).weight_norm()
            )

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (index + 1))
            self.resblocks.append(
                ParallelBlock(channels, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = post_activation()
        self.conv_post = FishConvNet(
            channels, 1, post_conv_kernel_size, stride=1
        ).weight_norm()
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.checkpointing = False

    def forward(self, x):
        x = self.conv_pre(x)

        for index in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[index](x)

            if self.training and self.checkpointing:
                x = checkpoint(
                    self.resblocks[index],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.resblocks[index](x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self):
        for upsample in self.ups:
            upsample.remove_parametrizations()
        for block in self.resblocks:
            block.remove_parametrizations()
        self.conv_pre.remove_parametrizations()
        self.conv_post.remove_parametrizations()


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = FishConvNet(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        input_tensor = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)
        x = self.drop_path(x)

        if apply_residual:
            x = input_tensor + x

        return x


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: int = 7,
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            FishConvNet(
                input_channels,
                dims[0],
                kernel_size=7,
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for index in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[index], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[index], dims[index + 1], kernel_size=1),
            )
            self.downsample_layers.append(mid_layer)

        self.stages = nn.ModuleList()
        dp_rates = [item.item() for item in torch.linspace(0, drop_path_rate, sum(depths))]

        current = 0
        for index in range(len(depths)):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[index],
                        drop_path=dp_rates[current + offset],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_size=kernel_size,
                    )
                    for offset in range(depths[index])
                ]
            )
            self.stages.append(stage)
            current += depths[index]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for index in range(len(self.downsample_layers)):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)

        return self.norm(x)


class FireflyArchitecture(DAC):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        quantizer: nn.Module,
        spec_transform: nn.Module,
    ):
        nn.Module.__init__(self)

        self.backbone = backbone
        self.head = head
        self.quantizer = quantizer
        self.spec_transform = spec_transform
        self.downsample_factor = math.prod(self.quantizer.downsample_factor)

    def encode(self, audios, audio_lengths):
        audios = audios.float()

        mels = self.spec_transform(audios)
        mel_lengths = audio_lengths // self.spec_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        encoded_features = self.backbone(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // self.downsample_factor

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths) -> torch.Tensor:
        mel_masks = sequence_mask(
            feature_lengths * self.downsample_factor,
            indices.shape[2] * self.downsample_factor,
        )
        mel_masks_float_conv = mel_masks[:, None, :].float()
        audio_lengths = (
            feature_lengths * self.downsample_factor * self.spec_transform.hop_length
        )

        audio_masks = sequence_mask(
            audio_lengths,
            indices.shape[2] * self.downsample_factor * self.spec_transform.hop_length,
        )
        audio_masks_float_conv = audio_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        x = self.head(z) * audio_masks_float_conv

        return x, audio_lengths

    def remove_parametrizations(self):
        if hasattr(self.backbone, "remove_parametrizations"):
            self.backbone.remove_parametrizations()

        if hasattr(self.head, "remove_parametrizations"):
            self.head.remove_parametrizations()

    @property
    def device(self):
        return next(self.parameters()).device
