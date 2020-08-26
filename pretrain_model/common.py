import torch
from torch import nn
from torch.nn.utils import spectral_norm
from typing import Tuple, Union


def compute_endpoint_error(pred, tgt):
    return torch.norm(pred-tgt, p=2, dim=1).sum()


def add_sn(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        return spectral_norm(m)
    else:
        return m


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu).to(mu.device)
    z = mu + std * eps
    return z


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False), Swish(),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias=False,)


def conv1x1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,)


def deconv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias=False,)


def deconv1x1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.ConvTranspose3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,)


class Chomp(nn.Module):
    def __init__(self, chomp_sizes: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]], nb_dims):
        super(Chomp, self).__init__()
        if isinstance(chomp_sizes, int):
            self.chomp_sizes = [chomp_sizes] * nb_dims
        else:
            self.chomp_sizes = chomp_sizes

        assert len(self.chomp_sizes) == nb_dims, "must enter a chomp size for each dim"
        self.nb_dims = nb_dims

    def forward(self, x):
        input_size = x.size()

        if self.nb_dims == 1:
            slice_d = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            return x[:, :, slice_d].contiguous()

        elif self.nb_dims == 2:
            if self.chomp_sizes[0] % 2 == 0:
                slice_h = slice(self.chomp_sizes[0] // 2, input_size[2] - self.chomp_sizes[0] // 2, 1)
            else:
                slice_h = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            if self.chomp_sizes[2] % 2 == 0:
                slice_w = slice(self.chomp_sizes[1] // 2, input_size[3] - self.chomp_sizes[1] // 2, 1)
            else:
                slice_w = slice(0, input_size[3] - self.chomp_sizes[1], 1)
            return x[:, :, slice_h, slice_w].contiguous()

        elif self.nb_dims == 3:
            slice_d = slice(0, input_size[2] - self.chomp_sizes[0], 1)
            if self.chomp_sizes[1] % 2 == 0:
                slice_h = slice(self.chomp_sizes[1] // 2, input_size[3] - self.chomp_sizes[1] // 2, 1)
            else:
                slice_h = slice(0, input_size[3] - self.chomp_sizes[1], 1)
            if self.chomp_sizes[2] % 2 == 0:
                slice_w = slice(self.chomp_sizes[2] // 2, input_size[4] - self.chomp_sizes[2] // 2, 1)
            else:
                slice_w = slice(0, input_size[4] - self.chomp_sizes[2], 1)
            return x[:, :, slice_d, slice_h, slice_w].contiguous()

        else:
            raise RuntimeError("Invalid number of dims")
