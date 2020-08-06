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
from torch.optim import Adam
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

        self.optim = None
        self.optim_schedule = None
        self._setup_optim()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs, comment=None):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"

        if comment is None:
            comment = ""
            for expt in self.datasets_dict.keys():
                comment += "{}+".format(expt)
            comment += 'lr:{:.1e}'.format(self.train_config.lr)
            comment += 'lr:{:.1e}'.format(self.train_config.batch_size)

        self.writer = SummaryWriter(
            pjoin(self.train_config.runs_dir, "{}_{}".format(
                comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            self.iteration(self.unsupervised_train_loader, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                save_model(self.model,
                           prefix='chkpt:{:d}'.format(epoch+1),
                           comment=comment)

    # TODO: adding reg remains
    def iteration(self, dataloader, epoch=0):
        cuml_loss_dict = {'kl': 0.0, 'recon': 0.0, 'tot': 0.0}

        max_num_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=max_num_batches)
        for i, data_tuple in pbar:
            batch_src, batch_tgt = _send_to_cuda(data_tuple, self.device)

            if torch.isnan(batch_src).sum().item():
                print("i = {}. nan encountered in inputs. moving on".format(i))
                continue

            z, mu, logvar, recon = self.model(batch_src)
            loss_dict = self.model.compute_loss(mu, logvar, recon, batch_tgt)

            if torch.isnan(loss_dict['tot']).sum().item():
                print("i = {}. nan encountered in loss. moving on".format(i))
                continue

            for k, v in loss_dict.items():
                cuml_loss_dict[k] += v.item()

            global_step = epoch * max_num_batches + i
            if (global_step + 1) % self.train_config.log_freq == 0:
                # add losses to writerreinfor
                for k, v in loss_dict.items():
                    self.writer.add_scalar("loss/{}".format(k), v.item() / len(batch_src), global_step)

                # add optim state to writer
                if self.train_config.optim_choice == 'adam_with_warmup':
                    self.writer.add_scalar('lr', self.optim_schedule.current_lr, global_step)
                else:
                    log_lamb_rs(self.optim, self.writer, global_step)

            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            for k, v in loss_dict.items():
                msg1 += "{}: {:.2e}, ".format(k, v.item() / len(batch_src))

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            final_loss = loss_dict['tot'] / len(batch_src)

            # backward and optimization only in train
            if self.train_config.optim_choice == 'adam_with_warmup':
                self.optim_schedule.zero_grad()
                final_loss.backward()
                self.optim_schedule.step_and_update_lr()
            else:
                self.optim.zero_grad()
                final_loss.backward()
                self.optim.step()

            if i + 1 == max_num_batches:
                msg0 = 'epoch {:d}, '.format(epoch)
                msg1 = ""
                for k, v in cuml_loss_dict.items():
                    msg1 += "avg {}: {:.3e}, ".format(k, v / max_num_batches / self.train_config.batch_size)
                desc2 = msg0 + msg1
                pbar.set_description(desc2)

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