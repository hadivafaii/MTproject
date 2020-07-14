import os
import yaml
from datetime import datetime
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .configuration import Config


def save_model(model, prefix=None, comment=None):
    config_dict = vars(model.config)
    to_hash_dict_ = dc(config_dict)
    hashed_info = str(hash(frozenset(sorted(to_hash_dict_))))

    if prefix is None:
        prefix = 'chkpt:0'

    save_dir = pjoin(
        model.config.base_dir,
        'saved_models',
        "[{}_{:s}]".format(comment, hashed_info),
        "{}_{:s}".format(prefix, datetime.now().strftime("[%Y_%m_%d_%H:%M]")))

    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), pjoin(save_dir, 'model.bin'))

    with open(pjoin(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f)


def load_model(model_id=-1, chkpt_id=-1, config=None, load_dir=None, verbose=True):
    from .model import MTLayer, MTNet

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

    if config.multicell:
        loaded_model = MTNet(config)
    else:
        loaded_model = MTLayer(config)
    loaded_model.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

    return loaded_model


def print_num_params(module: nn.Module):
    t = PrettyTable(['Module Name', 'Num Params'])

    for name, m in module.named_modules():
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if '.' not in name:
            if isinstance(m, type(module)):
                t.add_row(["Total", "{}".format(total_params)])
                t.add_row(['---', '---'])
            else:
                t.add_row([name, "{}".format(total_params)])
    print(t, '\n\n')


def create_reg_mat(time_lags, spatial_dim):
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

    reg_mats_dict = {
        'd2t': torch.tensor(temporal_mat, dtype=torch.float),
        'd2x': torch.tensor(spatial_mat, dtype=torch.float),
    }

    return reg_mats_dict


def compute_reg_loss(reg_vals, reg_mats, tensors):
    reg_losses = {}
    for reg_type, reg_val in reg_vals.items():
        w_size = tensors[reg_type].size()
        if len(w_size) == 2:
            w = tensors[reg_type]
        elif len(w_size) == 4:
            w = tensors[reg_type].flatten(end_dim=1).flatten(start_dim=1)
        else:
            raise RuntimeError("encountered tensor with size {}".format(w_size))

        # TODO: add a try except here, so whenever tensor is None it just skips
        loss = reg_val * ((w @ reg_mats[reg_type]) ** 2).sum()
        reg_losses.update({reg_type: loss})

    return reg_losses


def _get_nll(true, pred):
    _eps = np.finfo(np.float32).eps
    return np.sum(pred - true * np.log(pred + _eps), axis=0) / np.sum(true, axis=0)


def get_null_adj_nll(true, pred):
    nll = _get_nll(true, pred)

    r_0 = true.mean(0)
    null_nll = _get_nll(true, r_0)

    return -nll + null_nll


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "leaky_relu":
        return F.leaky_relu
    elif activation == "softplus":
        return F.softplus
    else:
        raise RuntimeError("activation should be relu/leaky_relu/softplus, not {}".format(activation))
