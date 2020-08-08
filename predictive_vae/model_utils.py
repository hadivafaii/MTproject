import os
import yaml
from datetime import datetime
from copy import deepcopy as dc
from prettytable import PrettyTable
from os.path import join as pjoin
import numpy as np
from time import time

import torch
from torch import nn
import torch.nn.functional as F

from .configuration import Config, TrainConfig
from utils.generic_utils import convert_time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def run_eval_loop(loaded_models: dict, batch_size: int = 512, base_dir: str = None):
    assert len(loaded_models) - 1 == max(list(loaded_models.keys())), "Not all models are loaded"

    from .model import MTNet
    from .training import MTTrainer

    config = Config() if base_dir is None else Config(base_dir=base_dir)
    base_trainer = MTTrainer(MTNet(config, verbose=False), TrainConfig(batch_size=batch_size))

    mean_train_nnll = np.zeros(len(loaded_models))
    mean_valid_nnll = np.zeros(len(loaded_models))
    median_train_nnll = np.zeros(len(loaded_models))
    median_valid_nnll = np.zeros(len(loaded_models))

    start = time()

    for chkpt, _model in sorted(loaded_models.items()):
        print('\n\n')
        print('-' * 40, "chkpt: {}".format(chkpt), '-' * 40)
        base_trainer.swap_model(_model)
        out_dict = base_trainer.evaluate_model()

        mean_train_nnll[chkpt] = np.mean(out_dict['train_nnll_all']) if len(out_dict['train_nnll_all']) else 0
        mean_valid_nnll[chkpt] = np.mean(out_dict['valid_nnll_all']) if len(out_dict['valid_nnll_all']) else 0
        median_train_nnll[chkpt] = np.median(out_dict['train_nnll_all']) if len(out_dict['train_nnll_all']) else 0
        median_valid_nnll[chkpt] = np.median(out_dict['valid_nnll_all']) if len(out_dict['valid_nnll_all']) else 0

    end = time()

    bst_mean_idx = np.argmax(mean_valid_nnll)
    bst_median_idx = np.argmax(median_valid_nnll)

    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(mean_train_nnll, label="mean train")
    plt.plot(mean_valid_nnll, label="mean valid")
    plt.plot([bst_mean_idx, bst_mean_idx],
             [min(min(mean_train_nnll), min(mean_valid_nnll)),
              max(max(mean_train_nnll), max(mean_valid_nnll))],
             ls='--', label="best idx: {}".format(bst_mean_idx))
    plt.ylim(0.0, 0.2)
    plt.legend()
    plt.grid()

    plt.subplot(122)
    plt.plot(median_train_nnll, label="median train")
    plt.plot(median_valid_nnll, label="median valid")
    plt.plot([bst_median_idx, bst_median_idx],
             [min(min(median_train_nnll), min(median_valid_nnll)),
              max(max(median_train_nnll), max(median_valid_nnll))],
             ls='--', label="best idx: {}".format(bst_mean_idx))
    plt.legend()
    plt.grid()
    plt.show()

    convert_time(end - start)

    results = {
        "mean_train_nnll": mean_train_nnll,
        "mean_valid_nnll": mean_valid_nnll,
        "median_train_nnll": median_train_nnll,
        "median_valid_nnll": median_valid_nnll,
    }

    return base_trainer, results


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


def load_model(keyword, chkpt_id=-1, config=None, verbose=False, base_dir='Documents/PROJECTS/MT_LFP'):
    from .model import PredictiveVAE

    _dir = pjoin(os.environ['HOME'], base_dir, 'saved_models')
    available_models = os.listdir(_dir)
    if verbose:
        print('Available models to load:\n', available_models)

    match_found = False
    model_id = -1
    for i, model_name in enumerate(available_models):
        if keyword in model_name:
            model_id = i
            match_found = True
            break

    if not match_found:
        raise RuntimeError("no match found for keyword")

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

    loaded_model = PredictiveVAE(config, verbose=verbose)
    loaded_model.load_state_dict(torch.load(pjoin(load_dir, 'model.bin')))

    chkpt = load_dir.split("/")[-1].split("_")[0]
    model_name = load_dir.split("/")[-2]
    metadata = {"chkpt": chkpt, "model_name": model_name}

    return loaded_model.eval(), metadata


def print_num_params(module: nn.Module):
    t = PrettyTable(['Module Name', 'Num Params'])

    for name, m in module.named_modules():
        total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if '.' not in name:
            if isinstance(m, type(module)):
                t.add_row(["{}".format(m.__class__.__name__), "{}".format(total_params)])
                t.add_row(['---', '---'])
            else:
                t.add_row([name, "{}".format(total_params)])
    print(t, '\n\n')


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
