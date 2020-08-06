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

from .model_utils import print_num_params, get_activation_fn


class PredictiveVAE(nn.Module):
    def __init__(self, config, verbose=True):
        super(PredictiveVAE, self).__init__()

        self.config = config

        self.encoder = RotationalConvEncoder(config, verbose=verbose)
        self.decoder = ConvDecoder(config, verbose=verbose)
        # self.decoder = FFDecoder(config, verbose=verbose)

        self.recon_criterion = nn.MSELoss(reduction="sum")
        self.init_weights()
        if verbose:
            print_num_params(self)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)

        return z, mu, logvar, x_recon.flatten(start_dim=1)

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * eps
        return z

    def compute_loss(self, mu, logvar, x_recon, x_true):
        recon_term = self.recon_criterion(x_recon, x_true)
        kl_term = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

        loss_dict = {
            "kl": kl_term,
            "recon": recon_term,
            "tot": recon_term + self.config.beta * kl_term,
        }
        return loss_dict

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FFDecoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(FFDecoder, self).__init__()

        self.linear1 = nn.Linear(config.hidden_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 2 * config.grid_size ** 2)

        self.activation = nn.LeakyReLU(config.leaky_negative_slope)
        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)

        return x.view(-1, self.linear3.out_features)


class ConvDecoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(ConvDecoder, self).__init__()

        self.init_grid_size = config.decoder_init_grid_size
        self.linear = nn.Linear(
            config.hidden_size, config.nb_decoder_units[0] * self.init_grid_size * self.init_grid_size, bias=True)

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


class RotationalConvEncoder(nn.Module):
    def __init__(self, config, verbose=True):
        super(RotationalConvEncoder, self).__init__()

        self.nb_levels = len(config.nb_conv_units) + 1

        self.rot_chomp3d = Chomp(chomp_sizes=[k - 1 for k in config.rot_kernel_size], nb_dims=3)
        self.chomp3d = Chomp(chomp_sizes=[k - 1 for k in config.conv_kernel_size], nb_dims=3)

        self.rot_conv3d = RotConv3d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=[k - 1 for k in config.rot_kernel_size],)

        self.nb_conv_units = [config.nb_rotations * config.nb_rot_kernels] + config.nb_conv_units

        convs_list = []
        downsamples_list = []
        activations_list = []
        dropouts_list = []
        poolers_list = []
        spatial_fcs_list = [nn.Linear(config.grid_size ** 2, config.nb_spatial_units[0], bias=True)]
        temporal_fcs_list = [nn.Linear(config.time_lags, config.nb_temporal_units[0], bias=True)]
        time_lags_list = [config.time_lags]
        spatial_dims_list = [config.grid_size]
        for i in range(1, self.nb_levels):
            in_channels = self.nb_conv_units[i-1]
            out_channels = self.nb_conv_units[i]

            convs_list.append(nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.conv_kernel_size,
                padding=[k - 1 for k in config.conv_kernel_size],))
            downsamples_list.append(nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None)
            activations_list.append(nn.LeakyReLU(negative_slope=config.leaky_negative_slope))

            dropouts_list.append(nn.Dropout(config.dropout))
            pool_size = [config.time_lags // (2 ** i), config.grid_size // (2 ** i), config.grid_size // (2 ** i)]
            poolers_list.append(nn.AdaptiveAvgPool3d(pool_size))
            spatial_fcs_list.append(nn.Linear(np.prod(pool_size[1:]), config.nb_spatial_units[i], bias=True))
            temporal_fcs_list.append(nn.Linear(pool_size[0], config.nb_temporal_units[i], bias=True))
            time_lags_list.append(pool_size[0])
            spatial_dims_list.append(pool_size[1])

        self.convs = nn.ModuleList(convs_list)
        self.downsamples = nn.ModuleList(downsamples_list)
        self.dropouts = nn.ModuleList(dropouts_list)
        self.poolers = nn.ModuleList(poolers_list)
        self.spatial_fcs = nn.ModuleList(spatial_fcs_list)
        self.temporal_fcs = nn.ModuleList(temporal_fcs_list)
        self.activations = nn.ModuleList(activations_list)

        if config.regularization is not None:
            self.regularizer = Regularizer(
                reg_values=config.regularization,
                time_lags_list=time_lags_list,
                spatial_dims_list=spatial_dims_list,)

        output_size_zip = zip(self.nb_conv_units, config.nb_spatial_units, config.nb_temporal_units)
        output_sizes = sum([np.prod(tup) for tup in output_size_zip])

        self.fc1 = nn.Linear(output_sizes, config.hidden_size, bias=True)
        self.fc2 = nn.Linear(output_sizes, config.hidden_size, bias=True)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=config.leaky_negative_slope)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=config.leaky_negative_slope)
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
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        outputs += (x,)

        x1 = self.spatial_fcs[0](x.flatten(start_dim=3))  # N x C x tau x nb_spatial_units[0]
        x1 = x1.permute(0, 1, 3, 2)  # N x C x nb_spatial_fcs x tau
        x1 = self.temporal_fcs[0](x1)  # N x C x nb_spatial_fcs x nb_temporal_units[0]
        x1 = x1.flatten(start_dim=1)  # N x C*nb_spatial_units[0]*nb_temporal_units[0]
        outputs_flat += (x1,)

        for i in range(self.nb_levels - 1):
            x_pool = self.poolers[i](outputs[i])
            x = self.chomp3d(self.convs[i](x_pool))
            res = x_pool if self.downsamples[i] is None else self.downsamples[i](x_pool)
            x = self.activations[i](x + res)
            x = self.dropouts[i](x)
            outputs += (x,)

            x1 = self.spatial_fcs[i+1](x.flatten(start_dim=3))  # N x C x tau x nb_spatial_units[i+1]
            x1 = x1.permute(0, 1, 3, 2)  # N x C x nb_spatial_fcs x tau
            x1 = self.temporal_fcs[i+1](x1)  # N x C x nb_spatial_fcs x nb_temporal_units[i+1]
            x1 = x1.flatten(start_dim=1)  # N x C*nb_spatial_units[i+1]*nb_temporal_units[i+1]
            outputs_flat += (x1,)

        x = torch.cat(outputs_flat, dim=-1)  # N x output_size
        x = self.leaky_relu2(x)
        x = self.dropout2(x)

        mu = self.fc1(x)
        logvar = self.fc2(x)

        return mu, logvar


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
