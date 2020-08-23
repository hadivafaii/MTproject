import os
import joblib
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime
from sklearn.metrics import r2_score
from prettytable import PrettyTable
from copy import deepcopy as dc

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adamax, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import create_datasets, normalize_fn, ReadoutDataset
from .model_utils import save_model, load_model, get_null_adj_nll
import sys; sys.path.append('..')
from utils.generic_utils import to_np


class Trainer:
    def __init__(self, model, train_config, seed=665, load_unsupervised=False):
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
        self.load_unsupervised = load_unsupervised
        self._setup_data()

        self.writer = None

        self.optim_pretrain = None
        self.optim_finetune = None
        self.optim_schedule_pretrain = None
        self.optim_schedule_finetune = None
        self._setup_optim()

        self.readout_train_loader = None
        self.readout_valid_loader = None

        print("\nTotal Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, nb_epochs, comment, mode='finetune'):
        assert isinstance(nb_epochs, (int, range)), "Please provide either range or int"
        assert mode in ['pretrain', 'finetune'], "wrong mode encountered"

        self.writer = SummaryWriter(
            pjoin(self.train_config.runs_dir, "{}_{}".format(
                comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        if self.train_config.beta_warmup_steps is None:
            self.train_config.beta_warmup_steps = len(self.unsupervised_train_loader) * nb_epochs // 3

        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            self.iteration(mode=mode, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch))
                save_model(self.model,
                           prefix='chkpt:{:d}'.format(epoch),
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
            msg0 = 'epoch {:d}. '.format(epoch)
            msg1 = ""
            global_step = epoch * max_num_batches + i

            if mode == 'pretrain':
                src, tgt = _send_to_cuda(data, self.device)

                if global_step > self.train_config.beta_warmup_steps:
                    current_beta = 1.0
                else:
                    current_beta = global_step / self.train_config.beta_warmup_steps
                self.model.update_beta(current_beta)
                self.writer.add_scalar("beta", self.model.beta, global_step)

                _, (kl_x, kl_xz, recon_loss, loss) = self.model(src, tgt)
                final_loss = loss / self.train_config.batch_size
                cuml_loss += final_loss.item()

                _check_for_nans(src, tgt, loss, global_step)

                if (global_step + 1) % self.train_config.log_freq == 0:
                    self.writer.add_scalar("kl_terms/kl_x", kl_x.item(), global_step)
                    self.writer.add_scalar("kl_terms/kl_xz", kl_xz.item(), global_step)
                    self.writer.add_scalar("losses/recon_loss", recon_loss.item() / self.train_config.batch_size, global_step)
                    self.writer.add_scalar("losses/tot_loss", final_loss.item(), global_step)

                msg1 += "recon_loss: {:3f}, ".format(recon_loss.item() / self.train_config.batch_size)
                msg1 += "tot_loss: {:3f}, ".format(final_loss.item())

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
                    msg1 += "{}: {:.3f}, ".format(k, v.item())
                msg1 += "tot: {:.3f}, ".format(final_loss.item())
                if (global_step + 1) % self.train_config.log_freq == 0:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar('loss/{}'.format(k), v.item(), global_step)

            else:
                raise RuntimeError("invalid mode value encountered: {}".format(mode))

            # if self.config.regularization is None:
            # msg1 += "tot: {:.3f}, ".format(final_loss.item())
            # else:
            # reg_losses_dict = self.model.readoud.regularizer.compute_reg_loss(
            # self.model.encoder.temporal_fc, self.model.encoder_stim.spatial_fcs)

            # total_reg_loss = sum(x for x in reg_losses_dict.values())
            # final_loss += total_reg_loss

            # msg1 += "tot reg: {:.2e}".format(total_reg_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            # backward and optimization
            if mode == 'pretrain':
                self.optim_pretrain.zero_grad()
                final_loss.backward()
                if global_step < self.train_config.beta_warmup_steps:
                    _ = clip_grad_norm_(self.model.parameters(), 0.25)
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

            # if (global_step + 1) % self.train_config.log_freq == 0:
            # self.writer.add_scalar("tot_loss", final_loss.item(), global_step)

        if mode == 'pretrain':
            self.optim_schedule_pretrain.step()
        elif mode == 'finetune':
            self.optim_schedule_finetune.step()

    def generante_prediction(self, mode='valid', batch_size=None):
        self.model.eval()

        if batch_size is None:
            batch_size = self.train_config.batch_size

        preds_dict = {}
        for expt in self.experiment_names:
            if mode == 'train':
                indxs = self.supervised_dataset.train_indxs[expt]
            elif mode == 'valid':
                indxs = self.supervised_dataset.valid_indxs[expt]
            else:
                raise ValueError("Invalid mode: {}".format(mode))

            src = []
            tgt = []
            for idx in indxs:
                src.append(
                    np.expand_dims(self.supervised_dataset.stim[expt][..., idx - self.config.time_lags: idx], axis=0))
                tgt.append(np.expand_dims(self.supervised_dataset.spks[expt][idx], axis=0))

            src = np.concatenate(src).astype(float)
            tgt = np.concatenate(tgt).astype(float)

            num_batches = int(np.ceil(len(indxs) / batch_size))
            pred_list, true_list = [], []
            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, len(indxs))

                src_slice = src[range(start, end)]
                original_shape = src_slice.shape
                src_slice_normalized = normalize_fn(src_slice.reshape(original_shape[0], -1), dim=-1)

                batch_src = torch.tensor(src_slice_normalized.reshape(original_shape))
                batch_tgt = torch.tensor(tgt[range(start, end)])
                batch_src, batch_tgt = _send_to_cuda((batch_src, batch_tgt), self.device)

                with torch.no_grad():
                    pred = self.model(batch_src, expt)

                pred_list.append(pred)
                true_list.append(batch_tgt)

            preds_dict.update({expt: (to_np(torch.cat(pred_list)), to_np(torch.cat(true_list)))})

        return preds_dict

    def create_readout_dataloaders(self, keyword, experiment, from_pretrained=True, batch_size=None, base_dir=None):
        if from_pretrained:
            pint("loading pretrained model")
            loaded_models = {}
            for i in range(500):
                try:
                    _model, metadata = load_model(keyword, i, verbose=False, base_dir=base_dir)
                    chkpt = int(metadata['chkpt'].split(':')[-1])
                    loaded_models.update({chkpt: _model})
                except IndexError:
                    continue

            mod = sorted(loaded_models.items())[-1][-1]
            if base_dir is not None:
                mod.config.base_dir = base_dir
            self.swap_model(mod)
        else:
            print("using randomly initialized model")

        self.model.eval()

        if batch_size is None:
            batch_size = self.train_config.batch_size

        output_train = self._xtract(keyword, experiment, "train", batch_size)
        output_valid = self._xtract(keyword, experiment, "valid", batch_size)

        self.readout_train_loader = DataLoader(
            dataset=ReadoutDataset(output_train),
            batch_size=self.train_config.batch_size,
            shuffle=True, drop_last=True,)
        self.readout_valid_loader = DataLoader(
            dataset=ReadoutDataset(output_valid),
            batch_size=self.train_config.batch_size,
            shuffle=False, drop_last=True,)

    def _xtract(self, keyword, experiment, mode, batch_size):
        _dir = pjoin(self.config.base_dir, "xtracted_features")
        available_model_featrures = os.listdir(_dir)

        for name in available_model_featrures:
            if keyword == name:
                break

        load_dir = pjoin(_dir, keyword, experiment)
        load_file = pjoin(load_dir, "output_{}.sav".format(mode))

        try:
            return joblib.load(load_file)

        except FileNotFoundError:
            msg = "\nno match found for:\nkeyword = {}\nexperiment = {}\nmode = {}\nbuilding the data now"
            print(msg.format(keyword, experiment, mode))

            if mode == "train":
                indxs_dict = self.supervised_dataset.train_indxs
            elif mode == "valid":
                indxs_dict = self.supervised_dataset.valid_indxs
            else:
                raise RuntimeError("invalid mode encountered: {}".format(mode))

            x_dict = {}
            tgt_dict = {}
            for expt in self.experiment_names:
                if expt != experiment:
                    continue
                src = []
                tgt = []
                for idx in indxs_dict[expt]:
                    src.append(np.expand_dims(
                        self.supervised_dataset.stim[expt][..., idx - self.config.time_lags: idx], axis=0))
                    tgt.append(np.expand_dims(self.supervised_dataset.spks[expt][idx], axis=0))

                # ndarr
                src = np.concatenate(src).astype(float)
                tgt = np.concatenate(tgt).astype(float)

                # tensor
                src, tgt = tuple(map(lambda z: torch.tensor(z), (src, tgt)))

                # cuda
                src = _send_to_cuda(src, self.device)[0]
                original_shape = src.shape
                src_normalized = normalize_fn(src.reshape(original_shape[0], -1), dim=-1)
                src = src_normalized.reshape(original_shape)

                num_batches = int(np.ceil(len(src) / batch_size))

                x1_list = []
                x2_list = []
                x3_list = []
                for b in range(num_batches):
                    start = b * batch_size
                    end = min((b + 1) * batch_size, len(src))

                    with torch.no_grad():
                        x1, x2, x3 = self.model.encoder(src[range(start, end)])

                    x1_list.append(x1.cpu())
                    x2_list.append(x2.cpu())
                    x3_list.append(x3.cpu())

                x_dict.update({expt: (src.cpu(), torch.cat(x1_list), torch.cat(x2_list), torch.cat(x3_list))})
                tgt_dict.update({expt: tgt.cpu()})

            output = {expt: (
                tuple(to_np(x) for x in x_dict[expt]),
                to_np(tgt_dict[expt]),
            ) for expt in self.experiment_names if expt == experiment}

            save_dir = pjoin(_dir, keyword, experiment)
            os.makedirs(save_dir, exist_ok=True)
            save_file = pjoin(save_dir, "output_{}.sav".format(mode))
            print("saving:\n {}".format(save_file))
            joblib.dump(output, save_file)
            print("done!\n")

            return output

    def evaluate_model(self, only_valid=True, print_table=False):
        train_nnll, train_r2 = {}, {}
        if not only_valid:
            train_preds_dict = self.generante_prediction(mode='train')
            for expt, (pred, true) in train_preds_dict.items():
                train_nnll.update({expt: get_null_adj_nll(pred, true)})
                train_r2.update({expt: r2_score(true, pred, multioutput='raw_values') * 100})
        else:
            train_preds_dict = {}

        valid_nnll, valid_r2 = {}, {}
        valid_preds_dict = self.generante_prediction(mode='valid')
        for expt, (pred, true) in valid_preds_dict.items():
            valid_nnll.update({expt: get_null_adj_nll(pred, true)})
            valid_r2.update({expt: r2_score(true, pred, multioutput='raw_values') * 100})

        if print_table:
            t = PrettyTable(['Experiment Name', 'Channel', 'Train NNLL', 'Valid NNLL', 'Train R^2', 'Valid R^2'])

            for expt, good_channels in self.config.useful_cells.items():
                for idx, cc in enumerate(good_channels):
                    t.add_row([
                        expt, cc,
                        np.round(train_nnll[expt][idx], 3),
                        np.round(valid_nnll[expt][idx], 3),
                        "{} {}".format(np.round(train_r2[expt][idx], 1), "%"),
                        "{} {}".format(np.round(valid_r2[expt][idx], 1), "%"),
                    ])

            print(t)

        train_nnll_all = [x for item in train_nnll.values() for x in item]
        train_r2_all = [x for item in train_r2.values() for x in item]
        if not only_valid:
            msg = 'train: \t avg nnll: {:.4f}, \t median nnll: {:.4f}, \t avg r2: {:.2f} {},'
            print(msg.format(np.mean(train_nnll_all), np.median(train_nnll_all), np.mean(train_r2_all), '%'))

        valid_nnll_all = [x for item in valid_nnll.values() for x in item]
        valid_r2_all = [x for item in valid_r2.values() for x in item]
        msg = 'valid: \t avg nnll: {:.4f}, \t median nnll: {:.4f}, \t avg r2: {:.2f} {},'
        print(msg.format(np.mean(valid_nnll_all), np.median(valid_nnll_all), np.mean(valid_r2_all), '%'))

        output = {
            "train_preds_dict": train_preds_dict,
            "valid_preds_dict": valid_preds_dict,
            "train_nnll": train_nnll, "valid_nnll": valid_nnll,
            "train_nnll_all": train_nnll_all, "valid_nnll_all": valid_nnll_all,
        }

        return output

    def swap_model(self, new_model):
        self.model = new_model.to(self.device).eval()
        self.config = new_model.config
        self._setup_optim()

    def _setup_data(self):
        supervised_dataset, unsupervised_dataset = create_datasets(
            self.config, self.train_config.xv_folds, self.rng, self.load_unsupervised)

        self.supervised_dataset = supervised_dataset
        self.supervised_train_loader = DataLoader(
            dataset=supervised_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True, drop_last=True,)

        if self.load_unsupervised:
            self.unsupervised_dataset = unsupervised_dataset
            self.unsupervised_train_loader = DataLoader(
                dataset=unsupervised_dataset,
                batch_size=self.train_config.batch_size,
                shuffle=True, drop_last=True,)

    def _setup_optim(self):
        self.optim_pretrain = AdamW(
            self.model.parameters(),
            lr=1e-2, weight_decay=0.0,
        )
        self.optim_schedule_pretrain = CosineAnnealingLR(
            self.optim_pretrain, T_max=self.train_config.scheduler_period, eta_min=1e-7)


def _send_to_cuda(data_tuple, device, dtype=torch.float32):
    if not isinstance(data_tuple, (tuple, list)):
        data_tuple = (data_tuple,)
    return tuple(map(lambda z: z.to(device=device, dtype=dtype, non_blocking=False), data_tuple))


def _check_for_nans(src, tgt, loss, global_step):
    if torch.isnan(src).sum().item():
        raise RuntimeError("global_step = {}. nan encountered in src. moving on".format(global_step))
    if torch.isnan(tgt).sum().item():
        raise RuntimeError("global_step = {}. nan encountered in tgt. moving on".format(global_step))
    if torch.isnan(loss).sum().item():
        raise RuntimeError("global_step = {}. nan encountered in loss. moving on".format(global_step))
