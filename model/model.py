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
from torch.nn.utils import weight_norm
from torch.optim import Adam

from .model_utils import create_reg_mat, print_num_params, get_activation_fn


class MTNet(nn.Module):
    def __init__(self, config):
        super(MTNet, self).__init__()
        assert config.multicell, "For multicell modeling only"

        self.config = config

        self.core = MTRotatioanlConvCore(config)
        self.readout = MTReadout(config, self.core.output_size)

        self.criterion = nn.PoissonNLLLoss(log_input=False)
        self.reg_mats_dict = create_reg_mat(config.time_lags, self.core.rot_conv2d.kernel_size[0])

        self.init_weights()
        print_num_params(self)

    def forward(self, x, experiment_name: str):
        out_core = self.core(x)
        out = self.readout(out_core, experiment_name)

        return out

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def extras_to_device(self, device):
        for reg_type, reg_mat in self.reg_mats_dict.items():
            self.reg_mats_dict[reg_type] = reg_mat.to(device)
        self.core.rot_conv2d.rot_mat_to_device(device)


class MTReadout(nn.Module):
    def __init__(self, config, hidden_size):
        super(MTReadout, self).__init__()

        self.config = config
        self.hidden_size = hidden_size

        load_dir = pjoin(config.base_dir, 'extra_info')
        nb_cells_dict = np.load(pjoin(load_dir, "nb_cells_dict.npy"), allow_pickle=True).item()
        ctrs_dict = np.load(pjoin(load_dir, "ctrs_dict.npy"), allow_pickle=True).item()
        if config.experiment_names is not None:
            self.nb_cells_dict = {expt: nb_cells_dict[expt] for expt in config.experiment_names}
            self.ctrs_dict = {expt: ctrs_dict[expt] for expt in config.experiment_names}
        else:
            self.nb_cells_dict = nb_cells_dict
            self.ctrs_dict = ctrs_dict

        self.norm = nn.LayerNorm(hidden_size, config.layer_norm_eps)

        # TODO: idea, you can replace a linear layer with FF and some dim reduction:
        #  hidden_dim -> readout_dim (e.g. 80)
        layers = {}
        for expt, nb_cells in self.nb_cells_dict.items():
            layers.update({expt: nn.Linear(hidden_size, nb_cells, bias=True)})
        self.layers = nn.ModuleDict(layers)

        self.activation = get_activation_fn(config.readout_activation_fn)
        print_num_params(self)

    def forward(self, x, experiment_name: str):
        # inout is N x hidden_size
        x = self.norm(x)
        y = self.layers[experiment_name](x)
        y = self.activation(y)

        return y


class MTRotatioanlConvCore(nn.Module):
    def __init__(self, config):
        super(MTRotatioanlConvCore, self).__init__()

        self.config = config

        self.temporal_fc = nn.Linear(config.time_lags, config.nb_temporal_kernels, bias=False)
        self.spatial_fc = weight_norm(nn.Linear(config.grid_size ** 2, config.nb_spatial_readouts, bias=False))
        self.nb_spatial_conv_channels = config.nb_rotations * config.nb_rot_kernels * config.nb_temporal_kernels
        self.nb_conv_layers = int(np.floor(np.log2(config.grid_size)))
        self.chomp = Chomp(chomp_size=config.spatial_kernel_size - 1, nb_dims=2)

        self.rot_conv2d = weight_norm(RotConv2d(
            in_channels=2,
            out_channels=config.nb_rot_kernels,
            nb_rotations=config.nb_rotations,
            kernel_size=config.spatial_kernel_size,
            padding=config.spatial_kernel_size - 1))

        convs_list = []
        poolers_list = []
        spatial_fcs_list = [weight_norm(nn.Linear(config.grid_size ** 2, config.nb_spatial_readouts, bias=False))]
        for i in range(1, self.nb_conv_layers + 1):
            convs_list.append(weight_norm(nn.Conv2d(
                in_channels=self.nb_spatial_conv_channels,
                out_channels=self.nb_spatial_conv_channels,
                kernel_size=config.spatial_kernel_size,
                padding=config.spatial_kernel_size - 1)))
            size = config.grid_size // (2 ** i)
            poolers_list.append(nn.AdaptiveAvgPool2d(size))
            spatial_fcs_list.append(weight_norm(nn.Linear(size ** 2, config.nb_spatial_readouts, bias=False)))

        self.convs = nn.ModuleList(convs_list)
        self.poolers = nn.ModuleList(poolers_list)
        self.spatial_fcs = nn.ModuleList(spatial_fcs_list)

        self.output_size = self.nb_spatial_conv_channels * config.nb_spatial_readouts * (self.nb_conv_layers + 1)
        self.activation = get_activation_fn(config.core_activation_fn)
        print_num_params(self)

    def forward(self, x):
        x = self._rot_st_fwd(x)
        x1 = self.spatial_fcs[0](x.flatten(start_dim=2)).flatten(start_dim=1)

        outputs = ()
        outputs_main = ()

        outputs += (x,)
        outputs_main += (x1,)

        for i in range(self.nb_conv_layers):
            x_pool = self.poolers[i](outputs[i])
            x = self.chomp(self.convs[i](x_pool))
            x = self.activation(x + x_pool)
            x1 = self.spatial_fcs[i+1](x.flatten(start_dim=2)).flatten(start_dim=1)
            outputs += (x,)
            outputs_main += (x1,)

        return torch.cat(outputs_main, dim=-1)

    def _rot_st_fwd(self, x):
        # temporal part
        x = self.temporal_fc(x)  # N x 2 x grd x grd x nb_temp_kernels
        x = x.permute(0, -1, 1, 2, 3)  # N x nb_temp_kernels x 2 x grd x grd

        # spatial part
        x = x.flatten(end_dim=1)  # N * nb_temp_ch x 2 x grd x grd
        x = self.chomp(self.rot_conv2d(x))  # N * nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.view(
             -1, self.config.nb_temporal_kernels,
             self.config.nb_rotations * self.config.nb_rot_kernels,
             self.config.grid_size, self.config.grid_size)  # N x nb_temp_ch x nb_rot * nb_rot_kers x grd x grd
        x = x.flatten(start_dim=1, end_dim=2)  # N x C x grd x grd

        return self.activation(x)


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
        self.rotation_mat = None
        self._build_rotation_mat()

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(out_channels * nb_rotations))

    def forward(self, x):
        augmented_weight = self._get_augmented_weight()
        return self._conv_forward(x, augmented_weight)

    def _build_rotation_mat(self):
        thetas = np.deg2rad(np.arange(0, 360, 360 / self.nb_rotations))
        c, s = np.cos(thetas), np.sin(thetas)
        self.rotation_mat = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.float).permute(2, 0, 1)

    def rot_mat_to_device(self, device):
        self.rotation_mat = self.rotation_mat.to(device)

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
    def __init__(self, chomp_size, nb_dims):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size
        self.nb_dims = nb_dims

    def forward(self, x):
        if self.nb_dims == 1:
            return x[:, :, :-self.chomp_size].contiguous()
        elif self.nb_dims == 2:
            return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()
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
