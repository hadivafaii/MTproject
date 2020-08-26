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

from .common import *
from .model_utils import print_num_params


class PredNVAE(nn.Module):
    def __init__(self, config, verbose=False):
        super(PredNVAE, self).__init__()

        self.beta = 0.0
        self.config = config

        self.encoder = Encoder(config, verbose=verbose)
        self.decoder = Decoder(config, verbose=verbose)

        # self.recon_criterion = nn.MSELoss(reduction="sum")
        self.init_weights()
        self.apply(add_sn)

        if verbose:
            print_num_params(self)

    def forward(self, src, tgt):
        (x1, x2, x3, z1), (mu_x, logvar_x) = self.encoder(src)
        (y1, y2, y3, z2), (mu_z, logvar_z), (mu_xz, logvar_xz) = self.decoder(z1, x2)

        kl_x, kl_xz, recon_loss, loss = self._compute_loss(
            y3, tgt, mu_x, mu_xz, logvar_z, logvar_x, logvar_xz)

        return y3, (kl_x, kl_xz, recon_loss, loss)

    def update_beta(self, new_beta):
        assert 0.0 <= new_beta <= 1.0, "beta must be in [0, 1] interval"
        self.beta = new_beta

    def _compute_loss(self, recon, tgt, mu_x, mu_xz, logvar_z, logvar_x, logvar_xz):
        kl_x = 0.5 * torch.sum(
            torch.pow(mu_x, 2) + torch.exp(logvar_x) - logvar_x - 1
        )
        kl_xz = 0.5 * torch.sum(
            torch.pow(mu_xz, 2) * torch.exp(-logvar_z) +
            torch.exp(logvar_xz) - logvar_xz - 1
        )

        kl_loss = self.beta * (kl_x + kl_xz)
        recon_loss = compute_endpoint_error(recon, tgt)
        loss = kl_loss + recon_loss

        return kl_x, kl_xz, recon_loss, loss

    def init_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """ Initialize the weights """
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        else:
            pass


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


class Decoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Decoder, self).__init__()

        self.z_dim = config.z_dim
        self.inplanes = config.nb_rot_kernels * config.nb_rotations * 2 ** config.nb_lvls

        self.init_size = tuple(config.decoder_init_grid_size)
        self.expand1 = nn.Sequential(
            deconv1x1x1(config.z_dim, self.inplanes),
            nn.BatchNorm3d(self.inplanes), Swish(),)
        self.layer1 = self._make_layer(self.inplanes // 2, blocks=1, stride=2)
        self.proj1 = nn.Sequential(
            nn.ConvTranspose3d(self.inplanes, self.inplanes, 2, bias=False),
            nn.BatchNorm3d(self.inplanes), Swish(),)

        self.intermediate_size = tuple(item * 2 for item in config.decoder_init_grid_size)
        self.swish = Swish()
        self.expand2 = nn.Sequential(
            deconv1x1x1(config.z_dim, self.inplanes),
            nn.BatchNorm3d(self.inplanes),)
        self.layer2 = self._make_layer(self.inplanes // 2, blocks=1, stride=2)
        self.proj2 = deconv1x1x1(self.inplanes, 2, bias=True)

        self.condition_z = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes * 2, config.z_dim * 2, bias=True),)
        self.condition_xz = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes * 4, config.z_dim * 2, bias=True),)

        if verbose:
            print_num_params(self)

    def forward(self, z1, x2):
        y1 = z1.view(-1, self.z_dim, 1, 1, 1)
        y1 = y1.expand(-1, self.z_dim, *self.init_size)
        y1 = self.expand1(y1)
        y1 = self.layer1(y1)
        y2 = self.proj1(y1)

        # side path
        mu_z, logvar_z = self.condition_z(y2).squeeze().chunk(2, dim=-1)
        xy = torch.cat([y2, x2], dim=1)
        mu_xz, logvar_xz = self.condition_xz(xy).squeeze().chunk(2, dim=-1)
        z2 = reparametrize(mu_z + mu_xz, logvar_z + logvar_xz)
        res = z2.view(-1, self.z_dim, 1, 1, 1)
        res = res.expand(-1, self.z_dim, *self.intermediate_size)
        res = self.expand2(res)

        # second layer
        y2 = self.swish(y2 + res)
        y2 = self.layer2(y2)
        y3 = self.proj2(y2)

        return (y1, y2, y3, z2), (mu_z, logvar_z), (mu_xz, logvar_xz)

    def generate(self, num_samples):
        self.eval()

        z1 = torch.randn((num_samples, self.z_dim))
        y1 = self.fc1(z1).view(num_samples, *self.init_size)
        y1 = self.layer1(y1)
        y2 = self.proj1(y1)

        mu_z, logvar_z = self.condition_z(y2).squeeze().chunk(2, dim=-1)
        z2 = reparametrize(mu_z, logvar_z)
        res = self.fc2(z2).view(num_samples, *self.intermediate_size)

        y2 = self.swish(y2 + res)
        y2 = self.layer2(y2)
        y3 = self.proj2(y2)

        return z1, z2, res, y3

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                deconv1x1x1(self.inplanes, planes, stride),
                nn.BatchNorm3d(planes),
            )

        layers = [DeConvBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DeConvBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class DeConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DeConvBlock, self).__init__()

        self.deconv1 = deconv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.swish1 = Swish()
        self.deconv2 = deconv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.swish2 = Swish()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.swish1(out)

        out = self.deconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish2(out)

        return out


class Encoder(nn.Module):
    def __init__(self, config, verbose=False):
        super(Encoder, self).__init__()

        self.rot_layer = RotationalConvBlock(config, verbose=verbose)
        self.inplanes = config.nb_rot_kernels * config.nb_rotations
        self.layer1 = self._make_layer(self.inplanes * 2, blocks=2, stride=2)
        self.layer2 = self._make_layer(self.inplanes * 2, blocks=2, stride=2)

        self.condition_x = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), conv1x1x1(self.inplanes, config.z_dim * 2, bias=True),)

        if verbose:
            print_num_params(self)

    def forward(self, x):
        x1 = self.rot_layer(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)

        mu_x, logvar_x = self.condition_x(x3).squeeze().chunk(2, dim=-1)
        z1 = reparametrize(mu_x, logvar_x)

        return (x1, x2, x3, z1), (mu_x, logvar_x)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes, stride),
                nn.BatchNorm3d(planes),
            )

        layers = [ConvBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ConvBlock(self.inplanes, planes))

        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ConvBlock, self).__init__()

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.swish1 = Swish()
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SELayer(planes, reduction=16)
        self.downsample = downsample
        self.swish2 = Swish()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.swish1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish2(out)

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
        self.swish1 = Swish()
        self.conv2 = conv3x3x3(nb_units, nb_units)
        self.bn2 = nn.BatchNorm3d(nb_units)
        self.se = SELayer(nb_units, reduction=8)
        self.swish2 = Swish()

        if verbose:
            print_num_params(self)

    def forward(self, x):
        # x : N x 2 x grd x grd x tau
        x = self.chomp3d(self.conv1(x))  # N x nb_rot_kers*nb_rot x grd x grd x tau
        x = self.bn1(x)
        x = self.swish1(x)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.se(out)
        out = self.swish2(out + x)

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
        return w.flatten(end_dim=1)


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
