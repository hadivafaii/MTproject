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

        self.encoder = Encoder(config, verbose=verbose)
        self.decoder = Decoder(config, verbose=verbose)
        # self.decoder = FFDecoder(config, verbose=verbose)
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
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
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

        self.linear1 = nn.Linear(config.z_dim, 4 * config.z_dim, bias=True)
        self.linear2 = nn.Linear(4 * config.z_dim, config.z_dim, bias=True)
        self.net = nn.Sequential(
            self.linear1, nn.ReLU(), nn.Dropout(config.dropout),
            self.linear2, nn.ReLU(), nn.Dropout(config.dropout))
        self.norm = nn.LayerNorm(config.z_dim, config.layer_norm_eps)

        layers = {}
        for expt, good_channels in config.useful_cells.items():
            layers.update({expt: nn.Linear(config.z_dim, len(good_channels))})
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


class Decoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Decoder, self).__init__()

        self.init_grid_size = config.decoder_init_grid_size
        self.linear = nn.Linear(
            config.z_dim, config.nb_decoder_units[0] * np.prod(self.init_grid_size), bias=True)

        layers = []
        for i in range(1, len(config.nb_decoder_units)):
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels=config.nb_decoder_units[i - 1],
                    out_channels=config.nb_decoder_units[i],
                    kernel_size=config.decoder_kernel_sizes[i - 1],
                    stride=config.decoder_strides[i - 1],
                    bias=False,),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
        self.net = nn.Sequential(*layers[:-2])

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(
            -1, self.linear.out_features // np.prod(self.init_grid_size),
            self.init_grid_size[0], self.init_grid_size[1], self.init_grid_size[2])
        x = self.net(x)

        return x.flatten(start_dim=1)


class Encoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Encoder, self).__init__()

        self.rot_layer = RotationalConvBlock(config, verbose=verbose)
        self.inplanes = config.nb_rot_kernels * config.nb_rotations
        self.layer1 = self._make_layer(ConvBlock, self.inplanes * 2, blocks=2, stride=2)
        self.layer2 = self._make_layer(ConvBlock, self.inplanes * 2, blocks=2, stride=2)

        # self.resnet = ResNet(config, verbose=verbose)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x1 = self.rot_layer(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)

        return x1, x2, x3

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes, stride),
                nn.BatchNorm3d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBlock, self).__init__()

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class RotationalConvBlock(nn.Module):
    def __init__(self, config, verbose=False):
        super(RotationalConvBlock, self).__init__()

        nb_units = config.nb_rot_kernels * config.nb_rotations
        padding = [k - 1 for k in config.rot_kernel_size]
        self.chomp3d = Chomp(chomp_sizes=padding, nb_dims=3)
        self.conv1 = RotConv3d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.rot_kernel_size,
            padding=padding,
            bias=False,)
        self.bn1 = nn.BatchNorm3d(nb_units)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(nb_units, nb_units)
        self.bn2 = nn.BatchNorm3d(nb_units)
        self.relu2 = nn.ReLU(inplace=True)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        # x : N x 2 x grd x grd x tau
        x = self.chomp3d(self.conv1(x))  # N x nb_rot_kers*nb_rot x grd x grd x tau
        x = self.bn1(x)
        x = self.relu1(x)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu2(out + x)

        return out


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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, dilation=dilation, bias=False,)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,)


class ConvNIM(nn.Module):
    def __init__(self, nb_kers, nb_tk, nb_sk, time_lags=12, rot_kernel_size=None):
        super(ConvNIM, self).__init__()

        if rot_kernel_size is None:
            rot_kernel_size = [3, 3, 3]

        padding = [k - 1 for k in rot_kernel_size]
        self.chomp3d = Chomp(chomp_sizes=padding, nb_dims=3)
        self.conv = RotConv3d(
            in_channels=2,
            out_channels=nb_kers,
            nb_rotations=8,
            kernel_size=rot_kernel_size,
            padding=padding,
            bias=False,)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_fc = nn.Linear(time_lags, nb_tk, bias=False)
        self.spatial_fc = nn.Linear(225, nb_sk, bias=False)

        self.layer = nn.Linear(nb_kers * 8 * nb_tk * nb_sk, 12, bias=True)

        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")
        self.softplus = nn.Softplus()

        print_num_params(self)

    def forward(self, x):
        # x : N x 2 x grd x grd x tau
        x = self.chomp3d(self.conv(x))  # N x nb_rot_kers*nb_rots x grd x grd x tau
        x = self.temporal_fc(x)  # N x nb_rot_kers*nb_rots x grd x grd x nb_tk
        x = self.relu(x)

        x = x.flatten(start_dim=-3, end_dim=-2).permute(0, 1, 3, 2)
        x = self.spatial_fc(x)  # N x nb_rot_kers*nb_rots x nb_tk x nb_sk
        x = x.flatten(start_dim=1)  # N x filters

        y = self.layer(x)
        y = self.softplus(y)

        return y


class DirSelectiveNIM(nn.Module):
    def __init__(self, nb_exc, nb_inh, nb_vel_tuning, nb_tk, nb_sk, time_lags=12):
        super(DirSelectiveNIM, self).__init__()

        self.dir_tuning = nn.Linear(2, nb_exc + nb_inh, bias=False)
        self.vel_tuning = nn.Sequential(conv1x1(1, 32), nn.ReLU(),
                                        conv1x1(32, 32), nn.ReLU(),
                                        conv1x1(32, nb_vel_tuning), nn.ReLU(),)

        self.temporal_kernels = nn.Linear(time_lags, nb_tk, bias=False)
        self.spatial_kernels = nn.Linear(15 ** 2, nb_sk, bias=True)

        self.layer = nn.Linear((nb_exc + nb_inh) * nb_vel_tuning * nb_sk * nb_tk, 12, bias=True)

        self.reg = Regularizer(reg_values={'d2t': 1e-4, 'd2x': 1e-3},
                               time_lags_list=[12],
                               spatial_dims_list=[15])
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

        # self._load_vel_tuning()
        print_num_params(self)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)  # N x tau x grd x grd x 2
        rho = torch.norm(x, dim=-1)

        # angular component
        f_theta = torch.exp(self.dir_tuning(x) / rho.masked_fill(rho == 0., 1e-8).unsqueeze(-1))  # N x tau x grd x grd x nb_exc+nb_inh

        # radial component
        original_shape = rho.size()  # N x tau x grd x grd
        rho = rho.flatten(end_dim=1).unsqueeze(1)  # N*tau x 1 x grd x grd
        f_r = self.vel_tuning(rho)  # N*tau x nb_vel_tuning x grd x grd
        f_r = f_r.view(
            original_shape[0], original_shape[1], -1, original_shape[-2], original_shape[-1])  # N x tau x nb_vel_tuning x grd x grd
        f_r = f_r.permute(0, 1, 3, 4, 2)  # N x tau x grd x grd x nb_vel_tuning

        # full subunit
        subunit = torch.einsum('btijk, btijl -> btijkl', f_theta, f_r)  # N x tau x grd x grd x nb_exc+nb_inh x nb_vel_tuning
        subunit = subunit.flatten(start_dim=-2)  # N x tau x grd x grd x (nb_exc+nb_inh)*nb_vel_tuning
        subunit = subunit.permute(0, -1, 1, 2, 3)  # N x (nb_exc+nb_inh)*nb_vel_tuning x tau x grd x grd

        # apply spatial and temporal kernels
        z = self.spatial_kernels(subunit.flatten(start_dim=-2))  # N x (nb_exc+nb_inh)*nb_vel_tuning x tau x nb_sk
        z = z.permute(0, 1, 3, 2)  # N x (nb_exc+nb_inh)*nb_vel_tuning x nb_sk x tau
        z = self.temporal_kernels(z)  # N x (nb_exc+nb_inh)*nb_vel_tuning x nb_sk x nb_tk
        z = z.flatten(start_dim=1)  # N x nb_filters
        z = self.relu(z)

        y = self.layer(z)
        y = self.softplus(y)

        return y

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
