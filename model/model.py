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

from .model_utils import create_reg_mat, print_num_params, get_activation_fn


class MTNet(nn.Module):
    def __init__(self, config, verbose=True):
        super(MTNet, self).__init__()
        assert config.multicell, "For multicell modeling only"

        self.config = config

        self.core = MTRotatioanlConvCoreNew(config, verbose=verbose)
        self.readout = MTReadout(config, verbose=verbose)

        # self.core = MTRotatioanlConvCore(config)
        # self.readout = MTReadout(config, self.core.output_size, verbose=verbose)

        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")
        self.init_weights()
        if verbose:
            print_num_params(self)

    def forward(self, x, experiment_name: str):
        out_core = self.core(x)
        out = self.readout(out_core, experiment_name)

        return out

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MTReadout(nn.Module):
    def __init__(self, config, verbose=True):
        super(MTReadout, self).__init__()

        self.config = config
        self.norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        # TODO: idea, you can replace a linear layer with FF and some dim reduction:
        #  hidden_dim -> readout_dim (e.g. 80)
        layers = {}
        for expt, good_channels in config.useful_cells.items():
            layers.update({expt: nn.Linear(config.hidden_size, len(good_channels), bias=True)})
        self.layers = nn.ModuleDict(layers)

        self.activation = get_activation_fn(config.readout_activation_fn)
        if verbose:
            print_num_params(self)

    def forward(self, x, experiment_name: str):
        # input is N x hidden_size
        x = self.norm(x)
        y = self.layers[experiment_name](x)  # N x nb_cells
        y = self.activation(y)

        return y


class MTRotationalDenseCore(nn.Module):
    def __init__(self, config, verbose=True):
        super(MTRotationalDenseCore, self).__init__()

        self.config = config

        self.chomp1 = Chomp(chomp_sizes=config.rot_kernel_size - 1, nb_dims=2)
        self.chomp2 = Chomp(chomp_sizes=config.spatial_kernel_size - 1, nb_dims=2)

        # 1st rotational spatio-temporal
        self.temporal_fc = nn.Linear(config.time_lags, config.nb_temporal_kernels, bias=False)
        self.rot_conv2d = weight_norm(RotConv2d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=config.rot_kernel_size - 1))

        nb_rot_out_channels = config.nb_rotations * config.nb_rot_kernels * config.nb_temporal_kernels
        self.nb_conv_layers = int(np.floor(np.log2(config.grid_size)))
        self.nb_conv_units = [nb_rot_out_channels] + [128, 256, 512]

        # deeper spatial part
        convs_list = []
        downsamples_list = []
        dropouts_list = []
        poolers_list = []
        self.pad_sizes = []
        for i in range(1, self.nb_conv_layers + 1):
            in_channels = self.nb_conv_units[i-1]
            out_channels = self.nb_conv_units[i]

            convs_list.append(weight_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.spatial_kernel_size,
                padding=config.spatial_kernel_size - 1)
            ))
            downsamples_list.append(nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None)

            dropouts_list.append(nn.Dropout(config.dropout))
            pool_size = config.grid_size // (2 ** i)
            poolers_list.append(nn.AdaptiveAvgPool2d(pool_size))
            self.pad_sizes.append((config.grid_size - pool_size) // 2)

        self.convs = nn.ModuleList(convs_list)
        self.downsamples = nn.ModuleList(downsamples_list)
        self.dropouts = nn.ModuleList(dropouts_list)
        self.poolers = nn.ModuleList(poolers_list)

        self.activation = get_activation_fn(config.core_activation_fn)
        self.dropout = nn.Dropout(config.dropout)
        if verbose:
            print_num_params(self)

    def forward(self, x):
        outputs = ()
        outputs_flat = ()

        x = self._rot_st_fwd(x)
        outputs += (x,)

        x1 = x.flatten(start_dim=2)  # N x C0 x grd**2
        outputs_flat += (x1,)

        for i in range(self.nb_conv_layers):
            x_pool = self.poolers[i](outputs[i])
            x = self.chomp2(self.convs[i](x_pool))
            res = x_pool if self.downsamples[i] is None else self.downsamples[i](x_pool)
            x = self.activation(x + res)
            x = self.dropouts[i](x)
            outputs += (x,)

            x1 = F.pad(x, (self.pad_sizes[i],) * 4)
            x1 = x1.flatten(start_dim=2)
            outputs_flat += (x1,)

        x = torch.cat(outputs_flat, dim=1)  # N x C x grd**2
        x = self.activation(x)

        return x

    def _rot_st_fwd(self, x):
        # temporal part
        x = self.temporal_fc(x)  # N x 2 x grd x grd x nb_temp_kernels
        x = x.permute(0, -1, 1, 2, 3)  # N x nb_temp_kernels x 2 x grd x grd

        # spatial part
        x = x.flatten(end_dim=1)  # N * nb_temp_ch x 2 x grd x grd
        x = self.chomp1(self.rot_conv2d(x))  # N * nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.view(
             -1, self.config.nb_temporal_kernels,
             self.config.nb_rotations * self.config.nb_rot_kernels,
             self.config.grid_size, self.config.grid_size)  # N x nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.flatten(start_dim=1, end_dim=2)  # N x C x grd x grd

        return self.activation(x)


class MTRotatioanlConvCore(nn.Module):
    def __init__(self, config):
        super(MTRotatioanlConvCore, self).__init__()

        config.nb_temporal_kernels = 2
        self.config = config

        self.chomp1 = Chomp(chomp_sizes=config.rot_kernel_size - 1, nb_dims=2)
        self.chomp2 = Chomp(chomp_sizes=config.spatial_kernel_size - 1, nb_dims=2)

        # 1st rotational spatio-temporal
        self.temporal_fc = nn.Linear(config.time_lags, config.nb_temporal_kernels, bias=False)
        self.rot_conv2d = weight_norm(RotConv2d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=config.rot_kernel_size - 1))

        self.nb_rot_out_channels = config.nb_rotations * config.nb_rot_kernels * config.nb_temporal_kernels
        self.nb_conv_layers = int(np.floor(np.log2(config.grid_size)))
        self.nb_conv_units = [self.nb_rot_out_channels] * 4
        # self.nb_conv_units = [self.nb_rot_out_channels] + [64, 64, 64]

        # deeper spatial part
        convs_list = []
        downsamples_list = []
        dropouts_list = []
        poolers_list = []
        spatial_fcs_list = [nn.Linear(config.grid_size ** 2, config.nb_spatial_units[0], bias=False)]
        for i in range(1, self.nb_conv_layers + 1):
            in_channels = self.nb_conv_units[i-1]
            out_channels = self.nb_conv_units[i]

            convs_list.append(weight_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.spatial_kernel_size,
                padding=config.spatial_kernel_size - 1)
            ))
            downsamples_list.append(nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None)

            dropouts_list.append(nn.Dropout(config.dropout))
            pool_size = config.grid_size // (2 ** i)
            poolers_list.append(nn.AdaptiveAvgPool2d(pool_size))
            spatial_fcs_list.append(nn.Linear(pool_size ** 2, config.nb_spatial_units[i], bias=False))

        self.convs = nn.ModuleList(convs_list)
        self.downsamples = nn.ModuleList(downsamples_list)
        self.dropouts = nn.ModuleList(dropouts_list)
        self.poolers = nn.ModuleList(poolers_list)
        self.spatial_fcs = nn.ModuleList(spatial_fcs_list)

        self.output_size = sum([np.prod(tup) for tup in zip(self.nb_conv_units, config.nb_spatial_units)])
        self.activation = get_activation_fn(config.core_activation_fn)
        self.dropout = nn.Dropout(config.dropout)
        print_num_params(self)

    def forward(self, x):
        outputs = ()
        outputs_flat = ()

        x = self._rot_st_fwd(x)
        outputs += (x,)

        x1 = self.spatial_fcs[0](x.flatten(start_dim=2)).flatten(start_dim=1)
        outputs_flat += (x1,)

        for i in range(self.nb_conv_layers):
            x_pool = self.poolers[i](outputs[i])
            x = self.chomp2(self.convs[i](x_pool))
            res = x_pool if self.downsamples[i] is None else self.downsamples[i](x_pool)
            x = self.activation(x + res)
            x = self.dropouts[i](x)
            outputs += (x,)

            x1 = self.spatial_fcs[i+1](x.flatten(start_dim=2)).flatten(start_dim=1)
            outputs_flat += (x1,)

        x = torch.cat(outputs_flat, dim=-1)
        # x = self.activation(x)

        return x

    def _rot_st_fwd(self, x):
        # temporal part
        x = self.temporal_fc(x)  # N x 2 x grd x grd x nb_temp_kernels
        x = x.permute(0, -1, 1, 2, 3)  # N x nb_temp_kernels x 2 x grd x grd

        # spatial part
        x = x.flatten(end_dim=1)  # N * nb_temp_ch x 2 x grd x grd
        x = self.chomp1(self.rot_conv2d(x))  # N * nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.view(
             -1, self.config.nb_temporal_kernels,
             self.config.nb_rotations * self.config.nb_rot_kernels,
             self.config.grid_size, self.config.grid_size)  # N x nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.flatten(start_dim=1, end_dim=2)  # N x C x grd x grd

        return self.activation(x)


class MTRotatioanlConvCoreNew(nn.Module):
    def __init__(self, config, verbose=True):
        super(MTRotatioanlConvCoreNew, self).__init__()

        self.config = config

        self.rot_chomp3d = Chomp(chomp_sizes=[k - 1 for k in config.rot_kernel_size], nb_dims=3)
        self.chomp3d = Chomp(chomp_sizes=[k - 1 for k in config.conv_kernel_size], nb_dims=3)

        self.rot_conv3d = RotConv3d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=[k - 1 for k in config.rot_kernel_size])

        self.nb_conv_units = [config.nb_rotations * config.nb_rot_kernels] + config.nb_conv_units

        convs_list = []
        downsamples_list = []
        dropouts_list = []
        poolers_list = []
        spatial_fcs_list = [nn.Linear(config.grid_size ** 2, config.nb_spatial_units[0], bias=True)]
        temporal_fcs_list = [nn.Linear(config.time_lags, config.nb_temporal_units[0], bias=True)]
        for i in range(1, config.nb_levels):
            in_channels = self.nb_conv_units[i-1]
            out_channels = self.nb_conv_units[i]

            convs_list.append(nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.conv_kernel_size,
                padding=[k - 1 for k in config.conv_kernel_size]))
            downsamples_list.append(nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None)

            dropouts_list.append(nn.Dropout(config.dropout))
            pool_size = [config.time_lags // (2 ** i), config.grid_size // (2 ** i), config.grid_size // (2 ** i)]
            poolers_list.append(nn.AdaptiveAvgPool3d(pool_size))
            spatial_fcs_list.append(nn.Linear(np.prod(pool_size[1:]), config.nb_spatial_units[i], bias=True))
            temporal_fcs_list.append(nn.Linear(pool_size[0], config.nb_temporal_units[i], bias=True))

        self.convs = nn.ModuleList(convs_list)
        self.downsamples = nn.ModuleList(downsamples_list)
        self.dropouts = nn.ModuleList(dropouts_list)
        self.poolers = nn.ModuleList(poolers_list)
        self.spatial_fcs = nn.ModuleList(spatial_fcs_list)
        self.temporal_fcs = nn.ModuleList(temporal_fcs_list)

        output_size_zip = zip(self.nb_conv_units, config.nb_spatial_units, config.nb_temporal_units)
        output_sizes = sum([np.prod(tup) for tup in output_size_zip])
        self.fc = nn.Linear(output_sizes, config.hidden_size, bias=True)

        self.activation = get_activation_fn(config.core_activation_fn)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        if verbose:
            print_num_params(self)

    def forward(self, x):
        outputs = ()
        outputs_flat = ()

        # x = self._rot_fwd(x)
        x = x.permute(0, 1, 4, 2, 3)  # N x 2 x tau x grd x grd
        x = self.rot_chomp3d(self.rot_conv3d(x))  # N x C x tau x grd x grd
        x = self.activation(x)
        x = self.dropout1(x)
        outputs += (x,)

        x1 = self.spatial_fcs[0](x.flatten(start_dim=3))  # N x C x tau x nb_spatial_units[0]
        x1 = x1.permute(0, 1, 3, 2)  # N x C x nb_spatial_fcs x tau
        x1 = self.temporal_fcs[0](x1)  # N x C x nb_spatial_fcs x nb_temporal_units[0]
        x1 = x1.flatten(start_dim=1)  # N x C*nb_spatial_units[0]*nb_temporal_units[0]
        outputs_flat += (x1,)

        for i in range(self.config.nb_levels - 1):
            x_pool = self.poolers[i](outputs[i])
            x = self.chomp3d(self.convs[i](x_pool))
            res = x_pool if self.downsamples[i] is None else self.downsamples[i](x_pool)
            x = self.activation(x + res)
            x = self.dropouts[i](x)
            outputs += (x,)

            x1 = self.spatial_fcs[i+1](x.flatten(start_dim=3))  # N x C x tau x nb_spatial_units[i+1]
            x1 = x1.permute(0, 1, 3, 2)  # N x C x nb_spatial_fcs x tau
            x1 = self.temporal_fcs[i+1](x1)  # N x C x nb_spatial_fcs x nb_temporal_units[i+1]
            x1 = x1.flatten(start_dim=1)  # N x C*nb_spatial_units[i+1]*nb_temporal_units[i+1]
            outputs_flat += (x1,)

        x = torch.cat(outputs_flat, dim=-1)  # N x output_size
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc(x)
        x = self.activation(x)

        return x


class RotConv3d(nn.Conv3d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            nb_rotations: int = 8,
            stride: int = 1,
            padding: int = None,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
    ):
        if padding is None:
            try:
                padding = kernel_size - 1
            except TypeError:
                padding = max(kernel_size) - 1

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


class RotConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            nb_rotations: int = 8,
            stride: int = 1,
            padding: int = None,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
    ):
        if padding is None:
            try:
                padding = kernel_size - 1
            except TypeError:
                padding = max(kernel_size) - 1

        super(RotConv2d, self).__init__(
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
        augmented_weight = self._get_augmented_weight()
        return self._conv_forward(x, augmented_weight)

    def _build_rotation_mat(self):
        thetas = np.deg2rad(np.arange(0, 360, 360 / self.nb_rotations))
        c, s = np.cos(thetas), np.sin(thetas)
        rotation_mat = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.float).permute(2, 0, 1)
        return rotation_mat

    def _get_augmented_weight(self):
        w = torch.einsum('jkn, inlm -> ijklm', self.rotation_mat, self.weight)
        w = w.flatten(end_dim=1)
        return w


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp(padding, nb_dims=1)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp(padding, nb_dims=1)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(config.nb_temporal_units)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else config.nb_temporal_units[i-1]
            out_channels = config.nb_temporal_units[i]
            layers += [TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=config.temporal_kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(config.temporal_kernel_size-1) * dilation_size,
                dropout=config.dropout),
            ]

        self.out_chanels = config.nb_temporal_units[-1]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


class MTLayer(nn.Module):
    def __init__(self, config):
        super(MTLayer, self).__init__()
        assert not config.multicell, "For single cell modeling only"
        self.config = config

        num_units = [1] + config.nb_vel_tuning_units + [1]
        layers = []
        for i in range(len(config.nb_vel_tuning_units) + 1):
            layers += [nn.Conv2d(num_units[i], num_units[i + 1], 1), nn.LeakyReLU()]

        self.vel_tuning = nn.Sequential(*layers)
        self.dir_tuning = nn.Linear(2, 1, bias=False)

        self.temporal_kernel = nn.Linear(config.time_lags, 1, bias=False)
        self.spatial_kernel = nn.Linear(config.grid_size ** 2, 1, bias=True)

        self.criterion = nn.PoissonNLLLoss(log_input=False)
        self.reg_mats_dict = create_reg_mat(config.time_lags, config.grid_size)
        self.activation = get_activation_fn(config.readout_activation_fn)

        self.init_weights()
        self._load_vel_tuning()
        print_num_params(self)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)  # N x tau x grd x grd x 2
        rho = torch.norm(x, dim=-1)

        # angular component
        f_theta = torch.exp(self.dir_tuning(x).squeeze(-1) / rho.masked_fill(rho == 0., 1e-8))

        # radial component
        original_shape = rho.size()  # N x tau x grd x grd
        rho = rho.flatten(end_dim=1).unsqueeze(1)  # N*tau x 1 x grd x grd
        f_r = self.vel_tuning(rho)
        f_r = f_r.squeeze(1).view(original_shape)

        # full subunit
        subunit = f_theta * f_r
        subunit = subunit.flatten(start_dim=2)  # N x tau x H*W

        # apply spatial and temporal kernels
        y = self.spatial_kernel(subunit).squeeze()  # N x tau
        y = self.temporal_kernel(y)  # N x 1
        y = self.activation(y)

        return y

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def _load_vel_tuning(self):
        _dir = pjoin(os.environ['HOME'], 'Documents/PROJECTS/MT_LFP/vel_dir_weights')
        try:
            print('[INFO] loading vel tuning identity weights')
            self.vel_tuning.load_state_dict(torch.load(pjoin(_dir, 'id_weights.bin')))
        except (FileNotFoundError, RuntimeError):
            print('[INFO] file does not exist, training from scratch')
            self._train_vel_tuning()
            os.makedirs(_dir, exist_ok=True)
            torch.save(self.vel_tuning.state_dict(), pjoin(_dir, 'id_weights.bin'))

    def _train_vel_tuning(self):
        if torch.cuda.is_available():
            self.cuda()

        tmp_optim = Adam(self.vel_tuning.parameters())
        loss_fn = nn.MSELoss()

        ratio = 10
        nb_epochs = 2000
        batch_size = 8192
        pbar = tqdm(range(nb_epochs))
        for epoch in pbar:
            tmp_data = torch.rand((batch_size, 1, self.config.grid_size, self.config.grid_size))
            tmp_data = tmp_data * ratio - 0.02
            tmp_data = tmp_data.cuda()

            pred = self.vel_tuning(tmp_data)
            loss = loss_fn(pred, tmp_data)

            tmp_optim.zero_grad()
            loss.backward()
            tmp_optim.step()

            pbar.set_description("epoch # {:d}, loss: {:.5f}".format(epoch, loss.item()))

        print('[INFO] training vel tuning identity weights done')

    def extras_to_device(self, device):
        for reg_type, reg_mat in self.reg_mats_dict.items():
            self.reg_mats_dict[reg_type] = reg_mat.to(device)

    def visualize(self, xv_nnll, xv_r2, save=False):
        dir_tuning = self.dir_tuning.weight.data.flatten().cpu().numpy()
        b_abs = np.linalg.norm(dir_tuning)
        theta = np.arccos(dir_tuning[1] / b_abs)

        tker = self.temporal_kernel.weight.data.flatten().cpu().numpy()
        sker = self.spatial_kernel.weight.data.view(self.config.grid_size, self.config.grid_size).cpu().numpy()

        if max(tker, key=abs) < 0:
            tker *= -1
            sker *= -1

        sns.set_style('dark')
        plt.figure(figsize=(16, 4))
        plt.subplot(121)
        t_rng = np.array([39, 36, 32, 27, 22, 15, 7, 0])
        plt.xticks(t_rng, (self.config.time_lags - t_rng - 1) * -self.config.temporal_res)
        plt.xlabel('Time (ms)', fontsize=25)
        plt.plot(tker)
        plt.grid()
        plt.subplot(122)
        plt.imshow(sker, cmap='bwr')
        plt.colorbar()

        plt.suptitle(
            '$\\theta_p = $ %.2f deg,     b_abs = %.4f     . . .     xv_nnll:  %.4f,       xv_r2:  %.2f %s'
            % (np.rad2deg(theta), b_abs, xv_nnll, xv_r2, '%'), fontsize=15)

        if save:
            result_save_dir = os.path.join(self.config.base_dir, 'results/PyTorch')
            os.makedirs(result_save_dir, exist_ok=True)
            save_name = os.path.join(
                result_save_dir,
                'DS_GLM_{:s}_{:s}.png'.format(self.config.experiment_names, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))
            )
            plt.savefig(save_name, facecolor='white')

        plt.show()
