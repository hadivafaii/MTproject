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
    def __init__(self, data_dict, time_lage, transform=None):

        self.data_dict = data_dict
        self.time_lags = time_lage
        self.transform = transform
        self.lengths = {expt: len(data['indxs']) for (expt, data) in self.data_dict.items()}

    def __len__(self):
        return max(list(self.lengths.values()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        source = {
            expt: data['stim'][..., data['indxs'][idx % self.lengths[expt]] - self.time_lags: data['indxs'][idx % self.lengths[expt]]] for
            (expt, data) in self.data_dict.items()}
        target = {
            expt: data['spks'][data['indxs'][idx % self.lengths[expt]]] for
            (expt, data) in self.data_dict.items()}

        if self.transform is not None:
            source = self.transform(source)

        return source, target


def normalize_fn(x):
    if isinstance(x, dict):
        return {k: v / v.std() for (k, v) in x.items()}
    else:
        return x / x.std()


def create_datasets(config, xv_folds, rng):
    data_dict_all = _load_data(config)
    train_data_all = {}
    valid_data_all = {}
    for expt, data_dict in data_dict_all.items():
        nt = len(data_dict['good_indxs'])
        train_inds, valid_inds = generate_xv_folds(nt, num_folds=xv_folds)
        rng.shuffle(train_inds)
        rng.shuffle(valid_inds)

        #train_stim = []
        #train_spks = []
        #for idx in data_dict['good_indxs'][train_inds]:
        #    train_stim.append(np.expand_dims(data_dict['stim'][..., idx - config.time_lags: idx], axis=0))
        #    train_spks.append(np.expand_dims(data_dict['spks'][idx], axis=0))
        #train_stim = np.concatenate(train_stim)
        #train_spks = np.concatenate(train_spks)

        #valid_stim = []
        #valid_spks = []
        #for idx in data_dict['good_indxs'][valid_inds]:
        #    valid_stim.append(np.expand_dims(data_dict['stim'][..., idx - config.time_lags: idx], axis=0))
        #    valid_spks.append(np.expand_dims(data_dict['spks'][idx], axis=0))
        #valid_stim = np.concatenate(valid_stim)
        #valid_spks = np.concatenate(valid_spks)

        train_data = {
            'stim': data_dict['stim'],
            'spks': data_dict['spks'],
            'indxs': data_dict['good_indxs'][train_inds],
        }
        valid_data = {
            'stim': data_dict['stim'],
            'spks': data_dict['spks'],
            'indxs': data_dict['good_indxs'][valid_inds],
        }

        train_data_all.update({expt: train_data})
        valid_data_all.update({expt: valid_data})

    if config.multicell:
        train_dataset = MTDataset(train_data_all, config.time_lags, normalize_fn)
        valid_dataset = MTDataset(valid_data_all, config.time_lags, normalize_fn)
    else:
        train_dataset = MTDataset(train_data_all, config.time_lags)
        valid_dataset = MTDataset(valid_data_all, config.time_lags)

    return train_dataset, valid_dataset, list(data_dict_all.keys())


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
