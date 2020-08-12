import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from typing import List, Tuple, Union, Dict


class Config:
    def __init__(
        self,
            useful_cells: Dict[str, list] = None,
            grid_size: int = 15,
            temporal_res: int = 25,
            time_lags: int = 12,

            hidden_size: int = 64,
            rot_kernel_size: Union[int, List[int]] = 3,
            nb_rot_kernels: int = 16,
            nb_rotations: int = 8,
            nb_temporal_units: int = 2,

            nb_readout_spatial_units: List[int] = None,
            nb_decoder_units: List[int] = None,
            decoder_kernel_sizes: List[int] = None,
            decoder_strides: List[int] = None,
            decoder_init_grid_size: int = 3,

            regularization: Dict[str, float] = None,
            dropout: float = 0.5,
            layer_norm_eps: float = 1e-12,
            base_dir: str = 'Documents/PROJECTS/MT_LFP',
            data_file: str = None,
    ):

        # generic configs
        self.grid_size = grid_size
        self.temporal_res = temporal_res
        self.time_lags = time_lags
        self.hidden_size = hidden_size

        # encoder
        if isinstance(rot_kernel_size, int):
            self.rot_kernel_size = [rot_kernel_size] * 3
        else:
            self.rot_kernel_size = rot_kernel_size
        self.nb_rot_kernels = nb_rot_kernels
        self.nb_rotations = nb_rotations
        self.nb_temporal_units = nb_temporal_units

        if nb_readout_spatial_units is None:
            self.nb_readout_spatial_units = [128, 8, 2]
        else:
            self.nb_readout_spatial_units = nb_readout_spatial_units

        # decoder
        if nb_decoder_units is None:
            self.nb_decoder_units = [256, 128, 64, 2]
        else:
            self.nb_decoder_units = nb_decoder_units
        if decoder_kernel_sizes is None:
            self.decoder_kernel_sizes = [3, 3, 1]
        else:
            self.decoder_kernel_sizes = decoder_kernel_sizes
        if decoder_strides is None:
            self.decoder_strides = [2, 2, 1]
        else:
            self.decoder_strides = decoder_strides
        self.decoder_init_grid_size = decoder_init_grid_size

        self.regularization = regularization
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        # dir configs
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        if data_file is None:
            self.data_file = pjoin(self.base_dir, 'python_processed', 'old_data_tres{:d}.h5'.format(temporal_res))
        else:
            self.data_file = data_file

        if useful_cells is None:
            self.useful_cells = self._load_cellinfo()
        else:
            self.useful_cells = useful_cells

    def _load_cellinfo(self):
        clu = pd.read_csv(pjoin(self.base_dir, "extra_info", "cellinfo.csv"))
        ytu = pd.read_csv(pjoin(self.base_dir, "extra_info", "cellinfo_ytu.csv"))

        clu = clu[np.logical_and(1 - clu.SingleElectrode, clu.HyperFlow)]
        ytu = ytu[np.logical_and(1 - ytu.SingleElectrode, ytu.HyperFlow)]

        useful_cells = {}

        for name in clu.CellName:
            useful_channels = []
            for i in range(1, 16 + 1):
                if clu[clu.CellName == name]["chan{:d}".format(i)].item():
                    useful_channels.append(i - 1)

            if len(useful_channels) > 1:
                useful_cells.update({name: useful_channels})

        for name in ytu.CellName:
            useful_channels = []
            for i in range(1, 24 + 1):
                if ytu[ytu.CellName == name]["chan{:d}".format(i)].item():
                    useful_channels.append(i - 1)

            if len(useful_channels) > 1:
                useful_cells.update({name: useful_channels})

        return useful_cells


class TrainConfig:
    def __init__(
            self,
            lr: Tuple[float, float] = (1e-4, 1e-2),
            weight_decay: Tuple[float, float] = (1e-4, 1e-3),
            scheduler_period: int = 10,
            batch_size: int = 1024,
            use_cuda: bool = True,
            log_freq: int = 10,
            chkpt_freq: int = 1,
            xv_folds: int = 5,
            runs_dir: str = 'Documents/MT/runs',
    ):

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_period = scheduler_period
        self.batch_size = batch_size

        self.use_cuda = use_cuda
        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.xv_folds = xv_folds
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
