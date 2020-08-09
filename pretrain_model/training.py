import os
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from copy import deepcopy as dc

import torch
from torch import nn
from torch.optim import Adam, Adamax
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from .optimizer import Lamb, log_lamb_rs, ScheduledOptim
from .dataset import create_datasets
from .model_utils import save_model

import sys; sys.path.append('..')
from utils.generic_utils import to_np, plot_vel_field


class Trainer:
    def __init__(self, model, train_config, seed=665):
        os.environ["SEED"] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = model.to(self.device)
        self.config = model.config
        self.train_config = train_config

        self.experiment_names = list(self.config.useful_cells.keys())
        self.supervised_dataset = None
        self.unsupervised_dataset = None
        self.supervised_train_loader = None
        self.unsupervised_train_loader = None
        self._setup_data()

        self.writer = None

        self.optim_pretrain = None
        self.optim_schedule_pretrain = None
        self._setup_optim_pretrain()

        self.optim_finetune = None
        self.optim_schedule_finetune = None
        self._setup_optim_finetune()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs, comment, mode='finetune'):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"
        assert mode in ['pretrain', 'finetune'], "wrong mode encountered"

        self.writer = SummaryWriter(
            pjoin(self.train_config.runs_dir, "{}_{}".format(
                comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            self.iteration(mode=mode, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                save_model(self.model,
                           prefix='chkpt:{:d}'.format(epoch+1),
                           comment=comment)

    def iteration(self, mode, epoch=0):
        if mode == 'pretrain':
            max_num_batches = len(self.unsupervised_train_loader)
            pbar = tqdm(enumerate(self.unsupervised_train_loader), total=max_num_batches)
        elif mode == 'finetune':
            max_num_batches = len(self.supervised_train_loader)
            pbar = tqdm(enumerate(self.supervised_train_loader), total=max_num_batches)
        else:
            raise RuntimeError("invalid mode value encountered: {}".format(mode))

        cuml_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_loss = 0.0

        for i, data in pbar:
            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            global_step = epoch * max_num_batches + i

            if mode == 'pretrain':
                src, tgt = _send_to_cuda(data, self.device)
                pred_stim = self.model(src)
                final_loss = self.model.criterion_stim(pred_stim, tgt) / self.train_config.batch_size
                cuml_loss += final_loss.item()

            elif mode == 'finetune':
                batch_data_dict = {expt: tuple(d[expt] for d in data) for expt in self.experiment_names}
                loss_dict = {}
                for expt, data_tuple in batch_data_dict.items():
                    src, tgt = _send_to_cuda(data_tuple, self.device)
                    pred_spks = self.model(src, expt)
                    loss = self.model.criterion_spks(pred_spks, tgt) / self.train_config.batch_size
                    loss_dict.update({expt: loss})
                    cuml_loss_dict[expt] += loss.item()

                final_loss = sum(x for x in loss_dict.values()) / len(loss_dict)

                for k, v in loss_dict.items():
                    msg1 += "{}: {:.3f}, ".format(k, v)
            else:
                raise RuntimeError("invalid mode value encountered: {}".format(mode))

            if self.config.regularization is None:
                msg1 += "tot: {:.3f}, ".format(final_loss.item())

            else:
                reg_losses_dict = self.model.readoud.regularizer.compute_reg_loss(
                    self.model.encoder.temporal_fc, self.model.encoder_stim.spatial_fcs)

                total_reg_loss = sum(x for x in reg_losses_dict.values())
                final_loss += total_reg_loss

                msg1 += "tot reg: {:.2e}".format(total_reg_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            # backward and optimization
            if mode == 'pretrain':
                self.optim_pretrain.zero_grad()
                final_loss.backward()
                self.optim_pretrain.step()
                self.writer.add_scalar('lr', self.optim_schedule_pretrain.get_last_lr()[0], global_step)
            elif mode == 'finetune':
                self.optim_finetune.zero_grad()
                final_loss.backward()
                self.optim_finetune.step()
                self.writer.add_scalar('lr/core', self.optim_schedule_finetune.get_last_lr()[0], global_step)
                self.writer.add_scalar('lr/readout', self.optim_schedule_finetune.get_last_lr()[1], global_step)

            if i + 1 == max_num_batches:
                msg0 = 'epoch {:d}, '.format(epoch)
                msg1 = ""
                if mode == 'pretrain':
                    msg1 += "avg loss: {:3f}".format(cuml_loss / max_num_batches)
                elif mode == 'finetune':
                    for k, v in cuml_loss_dict.items():
                        msg1 += "{}: {:.2f}, ".format(k, v / max_num_batches)
                    msg1 += "  . . .  avg loss: {:.3f}".format(
                        np.mean(list(cuml_loss_dict.values())) / max_num_batches)
                desc2 = msg0 + msg1
                pbar.set_description(desc2)

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("tot_loss", final_loss.item(), global_step)

        if mode == 'pretrain':
            self.optim_schedule_pretrain.step()
        elif mode == 'finetune':
            self.optim_schedule_finetune.step()

    def generante_prediction(self, mode='train'):
        self.model.eval()
        preds_dict = {}

        for expt in self.experiment_names:
            if mode == 'train':
                indxs = self.dataset.train_indxs[expt]
            elif mode == 'valid':
                indxs = self.dataset.valid_indxs[expt]
            else:
                raise ValueError("Invalid mode: {}".format(mode))

            src = []
            tgt = []
            for idx in indxs:
                src.append(np.expand_dims(self.dataset.stim[expt][..., idx - self.config.time_lags: idx], axis=0))
                tgt.append(np.expand_dims(self.dataset.spks[expt][idx], axis=0))

            src = np.concatenate(src).astype(float)
            tgt = np.concatenate(tgt).astype(float)

            num_batches = int(np.ceil(len(indxs) / self.train_config.batch_size))
            true_list = []
            pred_list = []
            for b in range(num_batches):
                start = b * self.train_config.batch_size
                end = min((b + 1) * self.train_config.batch_size, len(indxs))

                src_slice = src[range(start, end)]
                original_shape = src_slice.shape
                src_slice_normalized = normalize_fn(src_slice.reshape(original_shape[0], -1), dim=-1)

                data_tuple = (src_slice_normalized.reshape(original_shape), tgt[range(start, end)])
                data_tuple = tuple(map(lambda z: torch.tensor(z), data_tuple))

                batch_data_tuple = _send_to_cuda(data_tuple, self.device)
                # TODO: make this only for finetune maybe?
                with torch.no_grad():
                    pred = self.model(batch_data_tuple[0], expt)

                true_list.append(batch_data_tuple[1])
                pred_list.append(pred)

            true = torch.cat(true_list)
            pred = torch.cat(pred_list)
            preds_dict.update({expt: (to_np(true), to_np(pred))})

        return preds_dict

    def swap_model(self, new_model):
        self.model = new_model.to(self.device)
        self.config = new_model.config

    def _setup_optim(self):
        if self.train_config.optim_choice == 'lamb':
            self.optim = Lamb(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
                adam=False,
            )

        elif self.train_config.optim_choice == 'adam':
            self.optim = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )

        elif self.train_config.optim_choice == 'adamax':
            self.optim = Adamax(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )
            self.optim_schedule = CosineAnnealingLR(self.optim, T_max=10, eta_min=1e-7)

        elif self.train_config.optim_choice == 'adam_with_warmup':
            self.optim = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                betas=self.train_config.betas,
                weight_decay=self.train_config.weight_decay,
            )
            self.optim_schedule = ScheduledOptim(
                optimizer=self.optim,
                hidden_size=256,
                n_warmup_steps=self.train_config.warmup_steps,
            )

        else:
            raise ValueError("Invalid optimizer choice: {}".format(self.train_config.optim_chioce))

    def _setup_optim_pretrain(self):
        self.optim_pretrain = Adamax([
            {'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters()}],
            lr=1e-2, weight_decay=0.0,
        )
        self.optim_schedule_pretrain = CosineAnnealingLR(self.optim_pretrain, T_max=5, eta_min=1e-7)

    def _setup_optim_finetune(self):
        self.optim_finetune = Adamax([
            {'params': self.model.encoder.parameters(), 'lr': 1e-4},
            {'params': self.model.readout.parameters()}],
            lr=1e-2, weight_decay=0.0,
        )
        self.optim_finetune = CosineAnnealingLR(self.optim_finetune, T_max=10, eta_min=1e-7)

    def _setup_data(self):
        supervised_dataset, unsupervised_dataset = create_datasets(
            self.config, self.train_config.xv_folds, self.rng)
        self.supervised_dataset = supervised_dataset
        self.unsupervised_dataset = unsupervised_dataset

        self.supervised_train_loader = DataLoader(
            dataset=supervised_dataset,
            batch_size=self.train_config.batch_size,
            drop_last=True,)
        self.unsupervised_train_loader = DataLoader(
            dataset=unsupervised_dataset,
            batch_size=self.train_config.batch_size,
            drop_last=True,)


def _send_to_cuda(data_tuple, device, dtype=torch.float32):
    if not isinstance(data_tuple, (tuple, list)):
        data_tuple = (data_tuple,)
    return tuple(map(lambda z: z.to(device=device, dtype=dtype, non_blocking=False), data_tuple))
