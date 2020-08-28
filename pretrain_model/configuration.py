import os
import numpy as np
import pandas as pd
from os.path import join as pjoin
from typing import List, Tuple, Union, Dict


class Config:
    def __init__(
        self,
            predictive_model: bool = False,
            useful_cells: Dict[str, list] = None,
            temporal_res: int = 25,
            grid_size: int = 15,
            time_lags: int = 12,
            time_gals: int = 11,

            z_dim: int = 8,
            nb_lvls: int = 2,
            nb_rot_kernels: int = 4,
            nb_rotations: int = 8,
            rot_kernel_size: Union[int, List[int]] = 3,
            decoder_init_grid_size: List[int] = None,

            base_dir: str = 'Documents/PROJECTS/MT_LFP',
            data_file: str = None,
    ):

        # generic configs
        self.predictive_model = predictive_model
        self.temporal_res = temporal_res
        self.grid_size = grid_size
        self.time_lags = time_lags
        self.time_gals = time_gals
        self.z_dim = z_dim
        self.nb_lvls = nb_lvls

        # encoder
        self.nb_rot_kernels = nb_rot_kernels
        self.nb_rotations = nb_rotations
        if isinstance(rot_kernel_size, int):
            self.rot_kernel_size = [rot_kernel_size] * 3
        else:
            self.rot_kernel_size = rot_kernel_size

        # decoder
        if decoder_init_grid_size is None:
            self.decoder_init_grid_size = [4, 4, 3]
        else:
            self.decoder_init_grid_size = decoder_init_grid_size

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
            beta_warmup_steps: int = None,
            lr: float = 1e-2,
            weight_decay: float = 1e-4,
            scheduler_period: int = 5,
            batch_size: int = 1024,

            log_freq: int = 10,
            chkpt_freq: int = 1,
            xv_folds: int = 5,
            use_cuda: bool = True,
            runs_dir: str = 'Documents/MT/runs',
    ):

        self.beta_warmup_steps = beta_warmup_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_period = scheduler_period
        self.batch_size = batch_size

        self.log_freq = log_freq
        self.chkpt_freq = chkpt_freq
        self.xv_folds = xv_folds
        self.use_cuda = use_cuda
        self.runs_dir = pjoin(os.environ['HOME'], runs_dir)
