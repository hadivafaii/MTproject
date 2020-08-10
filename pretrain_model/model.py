import os
from datetime import datetime
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.optim import Adam

from .model_utils import print_num_params


class MTNet(nn.Module):
    def __init__(self, config, verbose=False):
        super(MTNet, self).__init__()

        self.config = config

        self.encoder = ConvEncoder(config, verbose=verbose)
        self.decoder = FFDecoder(config, verbose=verbose)
        # self.readout = MTReadout(config, verbose=verbose)
        self.readout = MTReadoutLite(config, verbose=verbose)

        self.criterion_stim = nn.MSELoss(reduction="sum")
        self.criterion_spks = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.init_weights()
        if verbose:
            print_num_params(self)

    def forward(self, src_stim, experiment_name: str = None):
        _, z = self.encoder(src_stim)

        if experiment_name is None:
            return self.decoder(z)
        else:
            # TODO: still not sure use z or the whole x_i gang for fine-tuning
            return self.readout(z, experiment_name)

    def init_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """ Initialize the weights """
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        else:
            pass


class MTReadoutFull(nn.Module):
    def __init__(self, config, verbose=False):
        super(MTReadoutFull, self).__init__()

        if verbose:
            print_num_params(self)

    def forward(self, z, experiment_name: str):

        return y


class MTReadoutLite(nn.Module):
    def __init__(self, config, verbose=False):
        super(MTReadoutLite, self).__init__()

        self.linear1 = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=True)
        self.net = nn.Sequential(
            self.linear1, nn.ReLU(), nn.Dropout(config.dropout),
            self.linear2, nn.ReLU(), nn.Dropout(config.dropout))
        self.norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        layers = {}
        for expt, good_channels in config.useful_cells.items():
            layers.update({expt: nn.Linear(config.hidden_size, len(good_channels))})
        self.layers = nn.ModuleDict(layers)

        self.dropout = nn.Dropout(config.dropout)
        self.softplus = nn.Softplus()

        if verbose:
            print_num_params(self)

    def forward(self, z, experiment_name: str):
        z = self.dropout(z)
        x = self.net(z)
        x = self.norm(x + z)

        y = self.layers[experiment_name](x)
        y = self.softplus(y)

        return y


# TODO: this is the original readout
class MTReadout(nn.Module):
    def __init__(self, config, verbose=False):
        super(MTReadout, self).__init__()

        self.linear = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True)
        self.norm = nn.LayerNorm(4 * config.hidden_size, config.layer_norm_eps)

        layers = {}
        for expt, good_channels in config.useful_cells.items():
            layers.update({expt: nn.Linear(4 * config.hidden_size, len(good_channels))})
        self.layers = nn.ModuleDict(layers)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        if verbose:
            print_num_params(self)

    def forward(self, z, experiment_name: str):
        x = self.relu(self.linear(z))
        x = self.norm(x)
        y = self.layers[experiment_name](x)
        y = self.softplus(y)

        return y


class FFDecoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(FFDecoder, self).__init__()

        self.linear1 = nn.Linear(config.hidden_size, 128)
        self.linear2 = nn.Linear(128, 2 * config.grid_size ** 2)

        self.relu = nn.ReLU()
        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(ConvDecoder, self).__init__()

        self.init_grid_size = config.decoder_init_grid_size
        self.linear = nn.Linear(2 * config.hidden_size,
                                config.nb_decoder_units[0] * self.init_grid_size * self.init_grid_size, bias=True)

        layers = []
        for i in range(1, len(config.nb_decoder_units)):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=config.nb_decoder_units[i - 1],
                    out_channels=config.nb_decoder_units[i],
                    kernel_size=config.decoder_kernel_sizes[i - 1],
                    stride=config.decoder_strides[i - 1],
                    bias=False,),
                nn.LeakyReLU(negative_slope=config.leaky_negative_slope),
                nn.Dropout(config.dropout),
            ])
        self.net = nn.Sequential(*layers)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(
            -1, self.linear.out_features // (self.init_grid_size ** 2),
            self.init_grid_size, self.init_grid_size)
        x = self.net(x)

        return x


class ConvEncoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(ConvEncoder, self).__init__()

        self.rot_layer = RotationalConvBlock(config, verbose=verbose)
        self.resnet = ResNet(config, verbose=verbose)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x0, x = self.rot_layer(x)
        x1, x2, x = self.resnet(x)

        return (x0, x1, x2), x


class ResNet(nn.Module):
    def __init__(self, config, verbose=False):
        super(ResNet, self).__init__()

        self.inplanes = config.nb_rot_kernels * config.nb_temporal_units * config.nb_rotations // 2

        self.layer1 = self._make_layer(BasicBlock, self.inplanes, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, self.inplanes * 2, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, config.hidden_size)

        if verbose:
            print_num_params(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x = self.avgpool(x2)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x1, x2, x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RotationalConvBlock(nn.Module):
    def __init__(self, config, verbose=False):
        super(RotationalConvBlock, self).__init__()

        padding = [k - 1 for k in config.rot_kernel_size]
        self.chomp3d = Chomp(chomp_sizes=padding, nb_dims=3)
        self.conv1 = RotConv3d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=padding,
            bias=False,)
        self.bn1 = nn.BatchNorm3d(config.nb_rot_kernels * config.nb_rotations)
        self.relu1 = nn.ReLU()
        self.temporal_fc = nn.Linear(config.time_lags, config.nb_temporal_units, bias=False)

        self.conv2 = conv1x1(config.nb_rot_kernels * config.nb_temporal_units * config.nb_rotations,
                             config.nb_rot_kernels * config.nb_temporal_units * config.nb_rotations // 2)
        self.bn2 = nn.BatchNorm2d(config.nb_rot_kernels * config.nb_temporal_units * config.nb_rotations // 2)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        # x : N x 2 x grd x grd x tau
        x = self.chomp3d(self.conv1(x))  # N x nb_rot_kers*nb_rot x grd x grd x tau
        x = self.bn1(x)
        # TODO: experiment with exp nonlinearity
        x = self.relu1(x)
        x = self.temporal_fc(x)  # N x nb_rot_kers*nb_rot x grd x grd x nb_t_units
        # TODO: should I take temporal fc before relu1 or after relu2?
        # TODO: should I use 2 relus here, or just 1?
        x = x.permute(0, 1, 4, 2, 3).flatten(start_dim=1, end_dim=2)  # N x nb_rot_kers*nb_rot*nb_t_units x grd x grd

        x = self.conv2(x)  # N x nb_rot_kers*nb_t_units x grd x grd
        x = self.bn2(x)
        x = self.relu2(x)

        x_pool = self.maxpool(x)  # N x nb_rot_kers*nb_rot*nb_t_units//2 x grd//2 x grd//2

        return x, x_pool


class RotConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            nb_rotations: int = 8,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
    ):
        super(RotConv3d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)

        self.nb_rotations = nb_rotations
        rotation_mat = self._build_rotation_mat()
        self.register_buffer('rotation_mat', rotation_mat)

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_channels * nb_rotations))

    def forward(self, x):
        # note: won't work when self.padding_mode != 'zeros'
        return F.conv3d(x, self._get_augmented_weight(), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def _build_rotation_mat(self):
        thetas = np.deg2rad(np.arange(0, 360, 360 / self.nb_rotations))
        c, s = np.cos(thetas), np.sin(thetas)
        rotation_mat = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.float).permute(2, 0, 1)
        return rotation_mat

    def _get_augmented_weight(self):
        w = torch.einsum('jkn, inlmo -> ijklmo', self.rotation_mat, self.weight)
        w = w.flatten(end_dim=1)
        return w


class Regularizer(nn.Module):
    def __init__(self, reg_values: dict, time_lags_list: list, spatial_dims_list: list):
        super(Regularizer, self).__init__()

        assert len(time_lags_list) == len(spatial_dims_list)

        self.time_lags_list = time_lags_list
        self.spatial_dims_list = spatial_dims_list

        self.lambda_d2t = reg_values['d2t']
        self.lambda_d2x = reg_values['d2x']

        self._register_reg_mats()

    def _register_reg_mats(self):
        d2t_reg_mats = []
        d2x_reg_mats = []

        for (time_lags, spatial_dim) in zip(self.time_lags_list, self.spatial_dims_list):
            temporal_mat = (
                    np.diag([1] * (time_lags - 1), k=-1) +
                    np.diag([-2] * time_lags, k=0) +
                    np.diag([1] * (time_lags - 1), k=1)
            )

            d1 = (
                    np.diag([1] * (spatial_dim - 1), k=-1) +
                    np.diag([-2] * spatial_dim, k=0) +
                    np.diag([1] * (spatial_dim - 1), k=1)
            )
            spatial_mat = np.kron(np.eye(spatial_dim), d1) + np.kron(d1, np.eye(spatial_dim))

            d2t_reg_mats.append(torch.tensor(temporal_mat, dtype=torch.float))
            d2x_reg_mats.append(torch.tensor(spatial_mat, dtype=torch.float))

        for i, mat in enumerate(d2t_reg_mats):
            self.register_buffer("d2t_mat_{:d}".format(i), mat)
        for i, mat in enumerate(d2x_reg_mats):
            self.register_buffer("d2x_mat_{:d}".format(i), mat)

    def compute_reg_loss(self, temporal_fcs: nn.ModuleList, spatial_fcs: nn.ModuleList):

        assert len(temporal_fcs) == len(self.time_lags_list)
        assert len(spatial_fcs) == len(self.spatial_dims_list)

        d2t_losses = []
        for i, layer in enumerate(temporal_fcs):
            mat = getattr(self, "d2t_mat_{:d}".format(i))
            d2t_losses.append(((layer.weight @ mat) ** 2).sum())
        d2t_loss = self.lambda_d2t * sum(item for item in d2t_losses)

        d2x_losses = []
        for i, layer in enumerate(spatial_fcs):
            w_size = layer.weight.size()
            if len(w_size) == 2:
                w = layer.weight
            elif len(w_size) == 4:
                w = layer.weight.flatten(end_dim=1).flatten(start_dim=1)
            else:
                raise RuntimeError("encountered tensor with size {}".format(w_size))
            mat = getattr(self, "d2x_mat_{:d}".format(i))
            d2x_losses.append(((w @ mat) ** 2).sum())

        d2x_loss = self.lambda_d2x * sum(item for item in d2x_losses)

        reg_losses = {'d2t': d2t_loss, 'd2x': d2x_loss}
        return reg_losses


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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias=False,)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,)
