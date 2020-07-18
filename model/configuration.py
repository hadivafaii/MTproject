import os
from os.path import join as pjoin
from typing import List


class Config:
    def __init__(
        self,
            experiment_names: List[str] = None,
            predictive_model: bool = False,
            grid_size: int = 15,
            temporal_res: int = 25,
            time_lags: int = 40,
            initializer_range: float = 0.01,
            multicell: bool = True,
            nb_vel_tuning_units: List[int] = None,
            core_activation_fn: str = 'leaky_relu',
            readout_activation_fn: str = 'softplus',
            spatial_kernel_size: int = 3,
            nb_rot_kernels: int = 20,
            nb_rotations: int = 8,
            nb_spatial_readouts: int = 100,
            nb_temporal_kernels: int = 3,
            temporal_kernel_size: int = 3,
            nb_temporal_units: int = None,
            # nb_spatial_blocks: int = 3,
            dropout: float = 0.1,
            layer_norm_eps=1e-12,
            base_dir: str = 'Documents/PROJECTS/MT_LFP',
            data_file: str = None,
    ):
        super(Config).__init__()

        # generic configs
        if experiment_names is not None and not isinstance(experiment_names, list):
            experiment_names = [experiment_names]
        self.experiment_names = experiment_names
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
        # self.nb_spatial_blocks = nb_spatial_blocks
        self.spatial_kernel_size = spatial_kernel_size
        self.nb_rot_kernels = nb_rot_kernels
        self.nb_rotations = nb_rotations
        self.nb_spatial_readouts = nb_spatial_readouts
        self.nb_temporal_kernels = nb_temporal_kernels
        self.temporal_kernel_size = temporal_kernel_size
        if nb_temporal_units is None:
            self.nb_temporal_units = [10, 10, 10]
        else:
            self.nb_temporal_units = nb_temporal_units
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        # dir configs
        self.base_dir = pjoin(os.environ['HOME'], base_dir)
        if data_file is None:
            self.data_file = pjoin(self.base_dir, 'python_processed', 'old_data_tres{:d}.h5'.format(temporal_res))
        else:
            self.data_file = data_file


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
