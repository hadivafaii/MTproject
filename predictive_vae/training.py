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

        # TODO: pretrain is not well defined right now

        self.optim = None
        self.optim_schedule = None
        self._setup_optim()

        # self.optim = None
        # self.optim_schedule = None
        # self._setup_optim()

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs, comment, mode='full'):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"
        assert mode in ['pretrain', 'full'], "wrong mode encountered"

        self.writer = SummaryWriter(
            pjoin(self.train_config.runs_dir, "{}_{}".format(
                comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            if mode == 'pretrain':
                self.iteration_pretrain(self.unsupervised_train_loader, epoch=epoch)
            elif mode == 'full':
                self.iteration(self.supervised_train_loader, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                save_model(self.model,
                           prefix='chkpt:{:d}'.format(epoch+1),
                           comment=comment)

    def iteration(self, dataloader, epoch=0):
        cuml_recon_spks_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_recon_stim_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_kl_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_tot_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_reg_loss = 0.0

        max_num_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=max_num_batches)
        for i, [src_stim_dict, tgt_stim_dict, src_spks_dict, tgt_spks_dict] in pbar:
            loss_dict = {}
            recon_spks_loss_dict = {}

            batch_data_dict = {expt: (src_stim_dict[expt],
                                      tgt_stim_dict[expt],
                                      src_spks_dict[expt],
                                      tgt_spks_dict[expt]) for expt in self.experiment_names}

            global_step = epoch * max_num_batches + i
            if global_step > self.config.beta_warmup_steps:
                current_beta = self.config.beta_range[1]
            else:
                current_beta = \
                    self.config.beta_range[0] + \
                    (self.config.beta_range[1] - self.config.beta_range[0]) * \
                    global_step / self.config.beta_warmup_steps
            self.model.update_beta(current_beta)
            self.writer.add_scalar("beta", self.model.beta, global_step)

            for expt, data_tuple in batch_data_dict.items():
                batch_data_tuple = _send_to_cuda(data_tuple, self.device)
                batch_src_stim, batch_tgt_stim, batch_src_spks, batch_tgt_spks = batch_data_tuple

                if torch.isnan(batch_src_stim).sum().item():
                    print("expt = {}. i = {}. nan encountered in inputs. moving on".format(expt, i))
                    continue
                elif torch.isnan(batch_tgt_stim).sum().item():
                    print("expt = {}. i = {}. nan encountered in targets. moving on".format(expt, i))
                    continue

                (_, mu, logvar), (recon_stim, recon_spks) = self.model(batch_src_stim, batch_src_spks, expt)
                vae_loss_dict = self.model.compute_loss(
                    mu, logvar, recon_stim, batch_tgt_stim, recon_spks, batch_tgt_spks)

                # if isinstance(self.model, nn.DataParallel):
                #    loss = self.model.module.criterion(pred, batch_tgt) / self.train_config.batch_size
                # else:
                #    loss = self.model.criterion(pred, batch_tgt) / self.train_config.batch_size

                if torch.isnan(vae_loss_dict['tot']).sum().item():
                    print("expt = {}. i = {}. nan encountered in loss. moving on".format(expt, i))
                    continue

                loss_dict.update({expt: vae_loss_dict['tot']})
                recon_spks_loss_dict.update({expt: vae_loss_dict['spks_recon'].item()})

                cuml_recon_spks_loss_dict[expt] += vae_loss_dict['spks_recon'].item()
                cuml_recon_stim_loss_dict[expt] += vae_loss_dict['stim_recon'].item()
                cuml_kl_loss_dict[expt] += vae_loss_dict['kl'].item()
                cuml_tot_loss_dict[expt] += vae_loss_dict['tot'].item()

            if (global_step + 1) % self.train_config.log_freq == 0:
                # add losses to writerreinfor
                for k, v in recon_spks_loss_dict.items():
                    self.writer.add_scalar("recon_spks_loss/{}".format(k), v, global_step)

            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            for k, v in recon_spks_loss_dict.items():
                msg1 += "{}: {:.3f}, ".format(k, v / self.train_config.batch_size)

            if self.config.regularization is None:
                final_loss = sum(x for x in loss_dict.values()) / (len(loss_dict) * self.train_config.batch_size)
                msg1 += "tot: {:.3f}, ".format(final_loss.item())

            else:
                reg_losses_dict = self.model.encoder_stim.regularizer.compute_reg_loss(
                    self.model.encoder_stim.temporal_fcs, self.model.encoder_stim.spatial_fcs)

                if (global_step + 1) % self.train_config.log_freq == 0:
                    for k, v in reg_losses_dict.items():
                        self.writer.add_scalar("reg_loss/{}".format(k), v.item(), global_step)

                total_reg_loss = sum(x for x in reg_losses_dict.values())
                cuml_reg_loss += total_reg_loss.item()

                total_loss = sum(x for x in loss_dict.values()) / (len(loss_dict) * self.train_config.batch_size)
                final_loss = total_loss + total_reg_loss

                msg1 += "tot reg: {:.2e}".format(total_reg_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            # backward and optimization
            self.optim.zero_grad()
            final_loss.backward()
            self.optim.step()

            # add optim state to writer
            self.writer.add_scalar('lr', self.optim_schedule.get_last_lr()[0], global_step)

            if i + 1 == max_num_batches:
                msg0 = 'epoch {:d}, '.format(epoch)
                msg1 = ""
                for k, v in cuml_recon_spks_loss_dict.items():
                    msg1 += "avg_recon_spks {}: {:.2f}, ".format(k, v / max_num_batches / self.train_config.batch_size)
                msg2 = " . . .  avg kl loss : {:.3f}, avg stim recon loss : {:.3f} avg loss: {:.3f}, avg reg loss: {:.2e}"
                msg2 = msg2.format(
                    np.mean(list(cuml_kl_loss_dict.values())) / max_num_batches / self.train_config.batch_size,
                    np.mean(list(cuml_recon_stim_loss_dict.values())) / max_num_batches / self.train_config.batch_size,
                    np.mean(list(cuml_tot_loss_dict.values())) / max_num_batches / self.train_config.batch_size,
                    cuml_reg_loss / max_num_batches,)
                desc2 = msg0 + msg1 + msg2
                pbar.set_description(desc2)

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("tot_loss", final_loss.item(), global_step)

        self.optim_schedule.step()

    # TODO: adding reg remains
    def iteration_pretrain(self, dataloader, epoch=0):
        cuml_loss_dict = {'kl': 0.0, 'recon': 0.0, 'tot': 0.0}

        max_num_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=max_num_batches)
        for i, data_tuple in pbar:
            batch_src, batch_tgt = _send_to_cuda(data_tuple, self.device)

            if torch.isnan(batch_src).sum().item():
                print("i = {}. nan encountered in inputs. moving on".format(i))
                continue

            z, mu, logvar, recon = self.model.vae(batch_src)
            loss_dict = self.model.vae.compute_loss(mu, logvar, recon, batch_tgt)

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

            # add optim state to writer
            self.writer.add_scalar('lr', self.optim_schedule.get_last_lr()[0], global_step)

            if i + 1 == max_num_batches:
                msg0 = 'epoch {:d}, '.format(epoch)
                msg1 = ""
                for k, v in cuml_loss_dict.items():
                    msg1 += "avg {}: {:.3e}, ".format(k, v / max_num_batches / self.train_config.batch_size)
                desc2 = msg0 + msg1
                pbar.set_description(desc2)

        if self.train_config.optim_choice == 'adamax':
            self.optim_schedule.step()

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
                with torch.no_grad():
                    if self.config.multicell:
                        pred = self.model(batch_data_tuple[0], expt)
                    else:
                        pred = self.model(batch_data_tuple[0])

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

    # TODO: it appeaers this is not needed at this point
    def _setup_optim_pretrain(self):
        self.optim_fine_tuning = Adam([
            {'params': self.model.encoder_stim.parameters()},
            {'params': self.model.decoder_stim.parameters()}],
            lr=1e-2, weight_decay=0.001,
        )
        self.optim_schedule_fine_tuning = CosineAnnealingLR(self.optim_fine_tuning, T_max=10, eta_min=1e-6)

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
