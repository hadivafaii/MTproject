import os
import yaml
from datetime import datetime
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.optim import Adam

from .configuration import Config


class RotConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: [int, Tuple[int]],
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
            self.bias = nn.Parameter(torch.Tensor(out_channels * nb_rotations))

    def forward(self, x):
        augmented_weight = self.get_augmented_weight()
        return self._conv_forward(x, augmented_weight)

    def _build_rotation_mat(self):
        thetas = np.deg2rad(np.arange(0, 360, 360 / self.nb_rotations))
        c, s = np.cos(thetas), np.sin(thetas)
        self.rotation_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.float).permute(2, 0, 1)

    def get_augmented_weight(self):
        w = [torch.einsum('ijk, klm -> ijlm', self.rotation_mat, self.weight[i]) for i in range(self.out_channels)]
        w = torch.cat(w)
        return w


# TODO
class SpatialBlock(nn.Module):
    pass


class SpatialConvNet(nn.Module):
    pass


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

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

        self.config = config

        num_units = [1] + config.nb_vel_tuning_units + [1]
        layers = []
        for i in range(len(config.nb_vel_tuning_units) + 1):
            layers += [nn.Conv2d(num_units[i], num_units[i + 1], 1), nn.ReLU()]

        self.vel_tuning = nn.Sequential(*layers)
        self.dir_tuning = nn.Linear(2, 1, bias=False)

        self.temporal_kernel = nn.Linear(config.time_lags, 1, bias=False)
        self.spatial_kernel = nn.Linear(config.grid_size ** 2, 1, bias=True)

        self.criterion = nn.PoissonNLLLoss(log_input=False)
        self.reg_mats_dict = self._create_reg_mat()
        self.activation = _get_activation_fn(config.activation_fn)

        self.init_weights()
        self.load_vel_tuning()
        self.print_num_params()

    def forward(self, x):
        x = x.permute(0, 1, 3, 4, 2)   # N x tau x grd x grd x 2
        rho = torch.norm(x, dim=-1)

        # angular component
        f_theta = torch.exp(self.dir_tuning(x).squeeze(-1) / rho.masked_fill(rho == 0., 1e-8))

        # radial component
        original_shape = rho.size()
        rho = rho.flatten(end_dim=1).unsqueeze(1)   # N*tau x 1 x H x W
        f_r = self.vel_tuning(rho)
        f_r = f_r.squeeze(1).view(original_shape)

        # full subunit
        subunit = f_theta * f_r
        subunit = subunit.flatten(start_dim=2)  # N x tau x H*W

        # apply spatial and temporal kernels
        y = self.temporal_kernel(subunit.permute(0, 2, 1)).squeeze(-1)
        y = self.spatial_kernel(y)
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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_vel_tuning(self):
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

    def print_num_params(self):
        t = PrettyTable(['Module Name', 'Num Params'])

        for name, m in self.named_modules():
            total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            if '.' not in name:
                if isinstance(m, type(self)):
                    t.add_row(["Total", "{}".format(total_params)])
                    t.add_row(['---', '---'])
                else:
                    t.add_row([name, "{}".format(total_params)])
        print(t, '\n\n')

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

    def save(self, prefix=None, comment=None):
        config_dict = vars(self.config)
        # data_config_dict = vars(self.data_config)

        to_hash_dict_ = dc(config_dict)
        # to_hash_dict_.update(data_config_dict)
        hashed_info = str(hash(frozenset(sorted(to_hash_dict_))))

        if prefix is None:
            prefix = 'chkpt:0'

        save_dir = pjoin(
            self.config.base_dir,
            'saved_models',
            "[{}_{:s}]".format(comment, hashed_info),
            "{}_{:s}".format(prefix, datetime.now().strftime("[%Y_%m_%d_%H:%M]")))

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.state_dict(), pjoin(save_dir, 'model.bin'))

        with open(pjoin(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)

        # with open(pjoin(save_dir, 'data_config.yaml'), 'w') as f:
        #    yaml.dump(data_config_dict, f)

    @staticmethod
    def load(model_id=-1, chkpt_id=-1, config=None, load_dir=None, verbose=True):
        if load_dir is None:
            _dir = pjoin(os.environ['HOME'], 'Documents/PROJECTS/MT_LFP/saved_models')
            available_models = os.listdir(_dir)
            if verbose:
                print('Available models to load:\n', available_models)
            model_dir = pjoin(_dir, available_models[model_id])
            available_chkpts = os.listdir(model_dir)
            if verbose:
                print('\nAvailable chkpts to load:\n', available_chkpts)
            load_dir = pjoin(model_dir, available_chkpts[chkpt_id])

        if verbose:
            print('\nLoading from:\n{}\n'.format(load_dir))

        if config is None:
            with open(pjoin(load_dir, 'config.yaml'), 'r') as stream:
                try:
                    config_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            config = Config(**config_dict)

        # if data_config is None:
        #    with open(pjoin(load_dir, 'data_config.yaml'), 'r') as stream:
        #        try:
        #            data_config_dict = yaml.safe_load(stream)
        #        except yaml.YAMLError as exc:
        #            print(exc)

        loaded_model = MTLayer(config)
        loaded_model.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

        return loaded_model

    def _create_reg_mat(self):
        temporal_mat = (
                np.diag([1] * (self.config.time_lags - 1), k=-1) +
                np.diag([-2] * self.config.time_lags, k=0) +
                np.diag([1] * (self.config.time_lags - 1), k=1)
        )

        d1 = (
                np.diag([1] * (self.config.grid_size - 1), k=-1) +
                np.diag([-2] * self.config.grid_size, k=0) +
                np.diag([1] * (self.config.grid_size - 1), k=1)
        )
        spatial_mat = np.kron(np.eye(self.config.grid_size), d1) + np.kron(d1, np.eye(self.config.grid_size))

        reg_mats_dict = {
            'd2t': torch.tensor(temporal_mat, dtype=torch.float),
            'd2x': torch.tensor(spatial_mat, dtype=torch.float),
        }

        return reg_mats_dict

    def reg_dicts_to_device(self, device):
        for reg_type, reg_mat in self.reg_mats_dict.items():
            self.reg_mats_dict[reg_type] = reg_mat.to(device)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "softplus":
        return F.softplus
    else:
        raise RuntimeError("activation should be relu/softplus, not {}".format(activation))
