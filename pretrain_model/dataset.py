import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from numpy.linalg import norm
from tqdm import tqdm
import joblib
import h5py

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')


def normalize_fn(x, dim=None):
    if isinstance(x, dict):
        return {k: (v - v.mean(axis=dim, keepdims=True)) / v.std(axis=dim, keepdims=True) for (k, v) in x.items()}
    else:
        return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


class ReadoutDataset(Dataset):
    def __init__(self, data_dict, keyword, experiment, transform=None):

        self.x = data_dict["sources"]
        self.spks = data_dict["target"]
        self.filters = data_dict["filters"]

        self.keyword = keyword
        self.experiment = experiment
        self.transform = transform

    def __len__(self):
        return len(self.spks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # normalizes the raw stim only
        if self.transform:
            src_x = (self.transform(self.x[0][idx]),) + tuple(x[idx] for x in self.x[1:])
        else:
            src_x = tuple(x[idx] for x in self.x)

        tgt = self.spks[idx]

        if self.filters is not None:
            fltr = self.filters[idx]
            return src_x, tgt, fltr
        else:
            return src_x, tgt, None


class NardinDataset(Dataset):
    def __init__(self, data_dict, time_lags, train=True, transform=None):

        assert not set(data_dict['train_indxs']).intersection(set(data_dict['valid_indxs'])), \
            "train/valid indices must be disjoint"

        self.stim = data_dict['stim']
        self.spks = data_dict['spks']
        self.filters = data_dict['filters']
        self.cell_ids = data_dict['cell_ids']

        if train:
            self.indxs = data_dict['train_indxs']
        else:
            self.indxs = data_dict['valid_indxs']

        self.time_lags = time_lags
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = self.indxs[idx]

        source = self.stim[..., i - self.time_lags: i]
        target = self.spks[i]
        filt = self.filters[i]

        if self.transform is not None:
            source = self.transform(source)

        return source, target, filt


class UnSupervisedDataset(Dataset):
    def __init__(self, data_dict, time_lags, time_gals, transform=None):

        self.stim = data_dict['stim']
        self.good_indxs = data_dict['good_indxs']
        self.train_indxs = data_dict['train_indxs']
        self.valid_indxs = data_dict['valid_indxs']

        assert not set(self.train_indxs).intersection(set(self.valid_indxs)), "train/valid indices must be disjoint"
        assert len(self.valid_indxs) + len(self.train_indxs) == len(self.good_indxs)

        self.time_lags = time_lags
        self.time_gals = time_gals
        self.transform = transform

    def __len__(self):
        return len(self.train_indxs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = self.good_indxs[self.train_indxs[idx]]

        source = self.stim[..., i - self.time_lags: i]
        target = self.stim[..., i: i + self.time_gals]

        if self.transform is not None:
            source = self.transform(source)
            target = self.transform(target)

        return source, target


class SupervisedDataset(Dataset):
    def __init__(self, data_dict, time_lags, transform=None):

        self.stim = {expt: data['stim'] for (expt, data) in data_dict.items()}
        self.spks = {expt: data['spks'] for (expt, data) in data_dict.items()}
        self.train_indxs = {expt: data['train_indxs'] for (expt, data) in data_dict.items()}
        self.valid_indxs = {expt: data['valid_indxs'] for (expt, data) in data_dict.items()}

        self.time_lags = time_lags
        self.transform = transform

        self.lengths = {expt: len(item) for (expt, item) in self.train_indxs.items()}

    def __len__(self):
        return max(list(self.lengths.values()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_dict = {expt: idx % length for (expt, length) in self.lengths.items()}

        source_stim = {
            expt: stim[..., self.train_indxs[expt][idx_dict[expt]] - self.time_lags: self.train_indxs[expt][idx_dict[expt]]] for
            (expt, stim) in self.stim.items()
        }
        target_spks = {
            expt: spks[self.train_indxs[expt][idx_dict[expt]]: self.train_indxs[expt][idx_dict[expt]]] for
            (expt, spks) in self.spks.items()
        }

        if self.transform is not None:
            source_stim = self.transform(source_stim)

        return source_stim, target_spks


def create_datasets(config, xv_folds, rng, load_unsupervised=False, load_processed=True):
    _dir = pjoin(config.base_dir, "pytorch_processed")
    files = os.listdir(_dir)

    if load_processed and len(files) == 3:
        print('processed data found: {}. loading . . .'.format(files))

        supervised_final = joblib.load(pjoin(_dir, "supervised.sav"))
        supervised_dataset = SupervisedDataset(supervised_final, config.time_lags, normalize_fn)

        nardin_final = joblib.load(pjoin(_dir, "nardin.sav"))
        nardin_dataset_train = NardinDataset(nardin_final, config.time_lags, True, normalize_fn)
        nardin_dataset_valid = NardinDataset(nardin_final, config.time_lags, False, normalize_fn)

        if load_unsupervised:
            unsupervised_final = joblib.load(pjoin(_dir, "unsupervised.sav"))
            unsupervised_dataset = UnSupervisedDataset(unsupervised_final, config.time_lags, config.time_gals, normalize_fn)
            return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, unsupervised_dataset
        else:
            return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, None

    supervised_data_dict, unsupervised_data_dict = _load_data(config)

    # supervised part
    supervised_final = {}
    for expt, data_dict in supervised_data_dict.items():
        nt = len(data_dict['good_indxs'])
        train_inds, valid_inds = generate_xv_folds(nt, num_folds=xv_folds)
        rng.shuffle(train_inds)
        data = {
            'stim': data_dict['stim'],
            'spks': data_dict['spks'],
            'train_indxs': data_dict['good_indxs'][train_inds],
            'valid_indxs': data_dict['good_indxs'][valid_inds],
        }
        supervised_final.update({expt: data})
    joblib.dump(supervised_final, pjoin(_dir, "supervised.sav"))
    supervised_dataset = SupervisedDataset(supervised_final, config.time_lags, normalize_fn)

    # nardin part
    nardin_final = _load_nardin_data(config)[0]
    joblib.dump(nardin_final, pjoin(_dir, "nardin.sav"))
    nardin_dataset_train = NardinDataset(nardin_final, config.time_lags, True, normalize_fn)
    nardin_dataset_valid = NardinDataset(nardin_final, config.time_lags, False, normalize_fn)

    # unsupervised part
    stim_all = []
    bad_indxs = []
    _eps = 1
    total_nt = 0
    for k, v in unsupervised_data_dict.items():
        bad_indxs.extend(range(total_nt, total_nt + config.time_lags + 1))

        nt = v.shape[-1]
        stim_norm = norm(v.reshape(-1, nt), axis=0)
        zero_norm_indices = np.where(stim_norm < _eps)[0]

        if len(zero_norm_indices) != 0:
            diff_mat = np.eye(len(zero_norm_indices)) - np.eye(len(zero_norm_indices), k=-1)
            boundary_indxs = np.where(diff_mat @ zero_norm_indices != 1)[0]

            for i in range(len(boundary_indxs) - 1):
                start = total_nt + zero_norm_indices[boundary_indxs[i]] - config.time_gals
                end = total_nt + zero_norm_indices[boundary_indxs[i + 1] - 1] + config.time_lags
                bad_indxs.extend(range(start, end))
            start = total_nt + zero_norm_indices[boundary_indxs[-1]] - config.time_gals
            end = total_nt + zero_norm_indices[-1] + config.time_lags
            bad_indxs.extend(range(start, end))

        total_nt += nt
        bad_indxs.extend(range(total_nt - config.time_gals - 1, total_nt))
        stim_all.append(v)

    good_indxs = set(range(total_nt)).difference(set(bad_indxs))
    stim_all = np.concatenate(stim_all, axis=-1)

    zero_norm_indices = np.where(norm(stim_all.reshape(-1, total_nt), axis=0) < _eps)[0]
    assert not set(zero_norm_indices).intersection(good_indxs), "good_indxs must exclude indices where ||stim|| = 0"
    assert stim_all.shape[-1] == total_nt

    train_inds, valid_inds = generate_xv_folds(len(good_indxs), num_folds=xv_folds, num_blocks=1)
    rng.shuffle(train_inds)

    unsupervised_final = {
        'stim': stim_all.astype(float),
        'good_indxs': list(good_indxs),
        'train_indxs': list(train_inds),
        'valid_indxs': list(valid_inds),
    }
    joblib.dump(unsupervised_final, pjoin(_dir, "unsupervised.sav"))
    unsupervised_dataset = UnSupervisedDataset(unsupervised_final, config.time_lags, config.time_gals, normalize_fn)

    return supervised_dataset, nardin_dataset_train, nardin_dataset_valid, unsupervised_dataset


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


def _load_nardin_data(config):
    path = pjoin(config.base_dir, "nardin", "python_processed")
    info = pd.read_pickle(pjoin(path, "info_w_ctrs_w_fr.pd"))
    info['2_name'] = np.array(info['2_name']).astype('str')

    threshold_fr = 3.0
    good_cells = list(info['4_fr'] > threshold_fr)
    info = info[good_cells]
    info = info.reset_index()

    unique_expt_names = list(np.unique(info["2_name"]))
    cell_ids = {}
    for expt in unique_expt_names:
        key = "nardin-{:s}".format(expt)
        indxs = list(info[info["2_name"] == expt].index)
        cell_ids.update({key: indxs})

    stim = np.transpose(np.load(os.path.join(path, "stim1.npy")), (3, 1, 2, 0))  # 2 x grd x grd x nt
    spks = np.load(os.path.join(path, "spks.npy"))[:, good_cells]
    filters = np.load(os.path.join(path, "data_filters.npy"))[:, good_cells]

    nt = stim.shape[-1]
    train_indxs, valid_indxs = generate_xv_folds(nt, num_folds=5, num_blocks=3)
    first_third = range(nt // 3)

    stim = stim[..., first_third]
    spks = spks[first_third]
    filters = filters[first_third]
    train_indxs = train_indxs[:len(train_indxs) // 3]
    valid_indxs = valid_indxs[:len(valid_indxs) // 3]

    train_indxs = [x for x in train_indxs if x > config.time_lags]
    valid_indxs = [x for x in valid_indxs if x > config.time_lags]

    data = {
        "stim": stim,
        "spks": spks,
        "filters": filters,
        "train_indxs": train_indxs,
        "valid_indxs": valid_indxs,
        "cell_ids": cell_ids,
    }

    return data, info


def _load_data(config):
    supervised_data_dict = {}
    unsupervised_data_dict = {}

    ff = h5py.File(config.data_file, 'r')
    for expt in tqdm(ff.keys()):
        grp = ff[expt]
        stim = np.transpose(np.array(grp['stim']), (3, 1, 2, 0))  # 2 x grd x grd x nt
        unsupervised_data_dict.update({expt: stim.astype(float)})

        if expt in config.useful_cells.keys():
            badspks = np.array(grp['badspks'])
            goodspks = 1 - badspks
            good_indxs = np.where(goodspks == 1)[0]
            good_indxs = good_indxs[good_indxs > config.time_lags]

            nt = stim.shape[-1]
            good_channels = config.useful_cells[expt]
            spks = np.zeros((nt, len(good_channels)))

            for i, cc in enumerate(good_channels):
                spks[:, i] = np.array(grp['ch_%d' % cc]['spks_%d' % cc]).squeeze()

            true_good_indxs = _refine_good_indices(stim, good_indxs, config.time_lags)

            _data = {
                'stim': stim.astype(float),
                'spks': spks.astype(float),
                'good_indxs': true_good_indxs.astype(int),
            }
            supervised_data_dict.update({expt: _data})

    ff.close()

    return supervised_data_dict, unsupervised_data_dict


def _refine_good_indices(stim, good_indxs, time_lags):
    _eps = 0.1
    nt = stim.shape[-1]
    stim_norm = norm(stim.reshape(-1, nt), axis=0)
    bad_indices = np.where(stim_norm < _eps)[0]

    true_good_indxs = []
    for i in good_indxs:
        if not set(range(i - time_lags, i + 1)).intersection(bad_indices):
            true_good_indxs.append(i)

    return np.array(true_good_indxs)
