import numpy as np
from tqdm import tqdm
import h5py

import torch
from torch.utils.data import Dataset


class MTDataset(Dataset):
    def __init__(self, config, experiment_name, data_dict):

        self.time_lags = config.time_lags
        self.predictive_model = config.predictive_model
        self.experiment_name = experiment_name
        self.data_dict = data_dict
        # self.epsilon = config.layer_norm_eps
        # self.data_dict = _load_data(config)
        # self.channel_sizes = {expt: spks.shape[1] for (expt, spks) in self.data_dict['spks_dict'].items()}

    def __len__(self):
        return len(self.data_dict['good_indxs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i = self.data_dict['good_indxs'][idx]
        source = self.data_dict['stim'][i - self.time_lags: i]
        if self.predictive_model:
            target = self.data_dict['stim'][i]
        else:
            target = self.data_dict['spks'][i]

        return source, target


def create_datasets(config):
    data_dict_all = _load_data(config)

    dataset_dicts = {}
    for expt, data_dict in data_dict_all.items():
        dataset_dicts.update({expt: MTDataset(config, expt, data_dict)})

    return dataset_dicts


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

        stim = np.array(grp['stim'])
        nt = len(stim)
        num_channels = np.array(grp['num_channels']).item()
        spks = np.zeros((nt, num_channels))
        for cc in range(num_channels):
            spks[:, cc] = np.array(grp['ch_%d' % cc]['spks_%d' % cc]).squeeze()

        _data = {
            'stim': stim.astype(float),
            'spks': spks.astype(int),
            'good_indxs': good_indxs.astype(int),
        }
        data_dict.update({key: _data})

    ff.close()

    return data_dict
