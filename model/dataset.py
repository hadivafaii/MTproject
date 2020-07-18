import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import h5py

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


class MTDataset(Dataset):
    def __init__(self, config, experiment_name, data_dict, normalize=True):

        self.time_lags = config.time_lags
        self.experiment_name = experiment_name
        self.data_dict = data_dict
        self.normalize = normalize

    def __len__(self):
        return len(self.data_dict['good_indxs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = self.data_dict['good_indxs'][idx]
        source = self.data_dict['stim'][..., i - self.time_lags: i]
        target = self.data_dict['spks'][i]

        # TODO: predictive model should have entirely different dateset structure

        if self.normalize:
            source /= source.std()

        return source, target


def create_datasets(config):
    data_dict_all = _load_data(config)
    if config.multicell:
        normalize = True
    else:
        normalize = False

    dataset_dicts = {}
    for expt, data_dict in data_dict_all.items():
        dataset_dicts.update({expt: MTDataset(config, expt, data_dict, normalize)})

    return dataset_dicts


def generate_xv_folds(nt, num_folds=5, num_blocks=3, which_fold=None):
    """Will generate unique and cross-validation indices, but subsample in each block
        NT = number of time steps
        num_folds = fraction of data (1/fold) to set aside for cross-validation
        which_fold = which fraction of data to set aside for cross-validation (default: middle of each block)
        num_blocks = how many blocks to sample fold validation from"""

    valid_inds = []
    nt_blocks = np.floor(nt / num_blocks).astype(int)
    block_sizes = np.zeros(num_blocks, dtype=int)
    block_sizes[range(num_blocks - 1)] = nt_blocks
    block_sizes[num_blocks - 1] = nt - (num_blocks - 1) * nt_blocks

    if which_fold is None:
        which_fold = num_folds // 2
    else:
        assert which_fold < num_folds, 'Must choose XV fold within num_folds = {}'.format(num_folds)

    # Pick XV indices for each block
    cnt = 0
    for bb in range(num_blocks):
        start = np.floor(block_sizes[bb] * (which_fold / num_folds))
        if which_fold < num_folds - 1:
            stop = np.floor(block_sizes[bb] * ((which_fold + 1) / num_folds))
        else:
            stop = block_sizes[bb]

        valid_inds = valid_inds + list(range(int(cnt + start), int(cnt + stop)))
        cnt = cnt + block_sizes[bb]

    valid_inds = np.array(valid_inds, dtype='int')
    train_inds = np.setdiff1d(np.arange(0, nt, 1), valid_inds)

    return list(train_inds), list(valid_inds)


def _load_data(config):
    data_dict = {}

    ff = h5py.File(config.data_file, 'r')
    for key in tqdm(ff.keys()):
        if config.experiment_names is not None and key not in config.experiment_names:
            continue

        grp = ff[key]

        badspks = np.array(grp['badspks'])
        goodspks = 1 - badspks
        good_indxs = np.where(goodspks == 1)[0]
        good_indxs = good_indxs[good_indxs > config.time_lags]

        stim = np.transpose(np.array(grp['stim']), (3, 1, 2, 0))   # 2 x grd x grd x nt
        nt = stim.shape[-1]
        num_channels = np.array(grp['num_channels']).item()
        spks = np.zeros((nt, num_channels))
        for cc in range(num_channels):
            spks[:, cc] = np.array(grp['ch_%d' % cc]['spks_%d' % cc]).squeeze()

        _eps = 0.1
        stim_norm = norm(stim.reshape(-1, nt), axis=0)
        bad_indices = np.where(stim_norm < _eps)[0]

        true_good_indxs = []
        for i in good_indxs:
            if not set(range(i - config.time_lags, i + 1)).intersection(bad_indices):
                true_good_indxs.append(i)
        true_good_indxs = np.array(true_good_indxs)

        _data = {
            'stim': stim.astype(float),
            'spks': spks.astype(float),
            'good_indxs': true_good_indxs.astype(int),
        }
        data_dict.update({key: _data})

    ff.close()

    return data_dict
