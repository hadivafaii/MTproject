import os
import yaml
from datetime import datetime
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .configuration import Config


class MTLayer(nn.Module):
    def __init__(self, config):
        super(MTLayer, self).__init__()

        self.config = config

        num_units = [1] + config.vel_tuning_num_units + [1]
        layers = []
        for i in range(len(config.vel_tuning_num_units) + 1):
            layers += [nn.Conv2d(num_units[i], num_units[i + 1], 1), nn.ReLU()]

        self.vel_tuning = nn.Sequential(*layers)
        self.dir_tuning = nn.Linear(2, 1, bias=False)

        self.temporal_kernel = nn.Linear(config.time_lags, 1, bias=False)
        self.spatial_kernel = nn.Linear(config.grid_size ** 2, 1, bias=True)

        self.criterion = nn.PoissonNLLLoss(log_input=False)
        self.activation = _get_activation_fn(config.activation_fn)

        self.init_weights()
        self.load_vel_tuning()
        self.print_num_params()

    def forward(self, x):
        x = x.squeeze(1)    # single cell (that dimension is num stim = num expts)
        rho = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)

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


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "softplus":
        return F.softplus
    else:
        raise RuntimeError("activation should be relu/softplus, not {}".format(activation))
