import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from typing import List, Tuple, Union, Dict


class Config:
    def __init__(
        self,
            useful_cells: Dict[str, list] = None,
            predictive_model: bool = False,
            grid_size: int = 15,
            temporal_res: int = 25,
            time_lags: int = 40,
            initializer_range: float = 0.01,
            multicell: bool = True,
            nb_vel_tuning_units: List[int] = None,
            core_activation_fn: str = 'leaky_relu',
            readout_activation_fn: str = 'softplus',
            nb_levels: int = 3,
            hidden_size: int = 32,
            rot_kernel_size: Union[int, List[int]] = 2,
            conv_kernel_size: Union[int, List[int]] = 2,
            nb_rot_kernels: int = 10,
            nb_rotations: int = 8,
            nb_conv_units: List[int] = None,
            nb_spatial_units: List[int] = None,
            nb_temporal_units: List[int] = None,
            # nb_temporal_kernels: int = 3,
            # nb_temporal_units: int = None,
            # nb_spatial_blocks: int = 3,
            dropout: float = 0.0,
            layer_norm_eps: float = 1e-12,
            base_dir: str = 'Documents/PROJECTS/MT_LFP',
            data_file: str = None,
    ):
        super(Config).__init__()

        # generic configs
        self.predictive_model = predictive_model
        self.grid_size = grid_size
        self.temporal_res = temporal_res
        self.time_lags = time_lags
        self.initializer_range = initializer_range

        self.multicell = multicell

        # single cell configs
        if nb_vel_tuning_units is None:
            self.nb_vel_tuning_units = [10, 10, 10, 10, 10]
        else:
            self.nb_vel_tuning_units = nb_vel_tuning_units

        # multicell or shared configs
        self.core_activation_fn = core_activation_fn
        self.readout_activation_fn = readout_activation_fn
        self.nb_levels = nb_levels
        self.hidden_size = hidden_size

        assert self.nb_levels - 1 <= int(np.floor(np.log2(self.grid_size)))

        if isinstance(rot_kernel_size, int):
            self.rot_kernel_size = [rot_kernel_size] * 3
        else:
            self.rot_kernel_size = rot_kernel_size
        if isinstance(conv_kernel_size, int):
            self.conv_kernel_size = [conv_kernel_size] * 3
        else:
            self.conv_kernel_size = conv_kernel_size

        self.nb_rot_kernels = nb_rot_kernels
        self.nb_rotations = nb_rotations

        if nb_conv_units is None:
            self.nb_conv_units = [128, 128, 128][:self.nb_levels - 1]
        else:
            self.nb_conv_units = nb_conv_units
        if nb_spatial_units is None:
            self.nb_spatial_units = [50, 10, 3, 1][:self.nb_levels]
        else:
            self.nb_spatial_units = nb_spatial_units
        if nb_temporal_units is None:
            self.nb_temporal_units = [2, 2, 1, 1][:self.nb_levels]
        else:
            self.nb_temporal_units = nb_temporal_units

        assert self.nb_levels == len(self.nb_conv_units) + 1 == len(self.nb_spatial_units) == len(self.nb_temporal_units)

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
            optim_choice='adam_with_warmup',
            regularization: dict = None,
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay: float = 0.01,
            warmup_steps: int = 1000,
            use_cuda: bool = True,
            log_freq: int = 10,
            chkpt_freq: int = 10,
            batch_size: int = 1024,
            xv_folds: int = 5,
            freeze_parameters_keywords: list = None,
            runs_dir: str = 'Documents/MT/runs',
    ):
        super(TrainConfig).__init__()

        if regularization is None:
            regularization = {'d2t': 1e-4, 'd2x': 1e-4}

        if freeze_parameters_keywords is None:
            freeze_parameters_keywords = []

        _allowed_optim_choices = ['lamb', 'adam', 'adam_with_warmup']
        assert optim_choice in _allowed_optim_choices, "Invalid optimzer choice, allowed options:\n{}".format(_allowed_optim_choices)

        self.optim_choice = optim_choice
        self.regularization = regularization
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_cuda = use_cuda
        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.batch_size = batch_size
        self.xv_folds = xv_folds
        self.freeze_parameters_keywords = freeze_parameters_keywords
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
