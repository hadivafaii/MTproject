import os
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime
from sklearn.metrics import r2_score
from prettytable import PrettyTable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from .dataset import create_datasets, normalize_fn
from .optimizer import Lamb, log_lamb_rs, ScheduledOptim
from .model_utils import save_model, compute_reg_loss, get_null_adj_nll

import sys; sys.path.append('..')
from utils.generic_utils import to_np, plot_vel_field


class MTTrainer:
    def __init__(self, model, train_config, seed=665):
        os.environ["SEED"] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        if train_config.use_cuda and torch.cuda.device_count() > 1:
            print("Using {:d} GPUs".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(model)
            self.model = self.model.to(self.device)
            self.config = self.model.module.config
        else:
            self.model = model.to(self.device)
            self.config = model.config

        self.train_config = train_config

        self.experiment_names = list(self.config.useful_cells.keys())
        self.dataset = None
        self.train_loader = None
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
            self.iteration(self.train_loader, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                # train_preds_dict, valid_preds_dict = self.evaluate_model()
                if isinstance(self.model, nn.DataParallel):
                    save_model(self.model.module,
                               prefix='chkpt:{:d}'.format(epoch+1),
                               comment=comment)
                else:
                    save_model(self.model,
                               prefix='chkpt:{:d}'.format(epoch+1),
                               comment=comment)

    def iteration(self, dataloader, epoch=0):
        cuml_loss_dict = {expt: 0.0 for expt in self.experiment_names}
        cuml_reg_loss = 0.0

        max_num_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=max_num_batches)
        for i, [src_dict, tgt_dict, _] in pbar:
            losses_dict = {}
            batch_data_dict = {expt: (src_dict[expt], tgt_dict[expt]) for expt in self.experiment_names}

            for expt, data_tuple in batch_data_dict.items():
                batch_data_tuple = _send_to_cuda(data_tuple, self.device)

                batch_src = batch_data_tuple[0]
                batch_tgt = batch_data_tuple[1]

                if torch.isnan(batch_src).sum().item():
                    print("expt = {}. i = {}. nan encountered in inputs. moving on".format(expt, i))
                    continue
                elif torch.isnan(batch_tgt).sum().item():
                    print("expt = {}. i = {}. nan encountered in targets. moving on".format(expt, i))
                    continue

                if self.config.multicell:
                    pred = self.model(batch_src, expt)
                else:
                    pred = self.model(batch_src)

                if isinstance(self.model, nn.DataParallel):
                    loss = self.model.module.criterion(pred, batch_tgt)
                else:
                    loss = self.model.criterion(pred, batch_tgt)

                if torch.isnan(loss).sum().item():
                    print("expt = {}. i = {}. nan encountered in loss. moving on".format(expt, i))
                    continue

                losses_dict.update({expt: loss})
                cuml_loss_dict[expt] += loss.item()

                global_step = epoch * max_num_batches + i
                if (global_step + 1) % self.train_config.log_freq == 0:
                    # add losses to writerreinfor
                    for k, v in losses_dict.items():
                        self.writer.add_scalar("loss/{}".format(k), v.item(), global_step)

                    # add optim state to writer
                    if self.train_config.optim_choice == 'adam_with_warmup':
                        self.writer.add_scalar('lr', self.optim_schedule.current_lr, global_step)
                    else:
                        log_lamb_rs(self.optim, self.writer, global_step)

            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            for k, v in losses_dict.items():
                msg1 += "{}: {:.3f}, ".format(k, v.item())

            if self.config.multicell:
                final_loss = sum(x for x in losses_dict.values()) / len(losses_dict)
                msg1 += "tot: {:.3f}, ".format(final_loss.item())

            else:
                reg_tensors = {
                    'd2t': self.model.temporal_kernel.weight,
                    'd2x': self.model.spatial_kernel.weight,
                }
                reg_losses_dict = compute_reg_loss(
                    reg_vals=self.train_config.regularization,
                    reg_mats=self.model.reg_mats_dict,
                    tensors=reg_tensors)

                global_step = epoch * max_num_batches + i
                if (global_step + 1) % self.train_config.log_freq == 0:
                    for k, v in reg_losses_dict.items():
                        self.writer.add_scalar("reg_loss/{}".format(k), v.item(), global_step)

                total_reg_loss = sum(x for x in reg_losses_dict.values()) / len(reg_losses_dict)
                cuml_reg_loss += total_reg_loss.item()

                total_loss = sum(x for x in losses_dict.values()) / len(losses_dict)
                final_loss = total_loss + total_reg_loss

                msg1 += "tot reg: {:.2e}".format(total_reg_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            # backward and optimization only in train
            if self.train_config.optim_choice == 'adam_with_warmup':
                self.optim_schedule.zero_grad()
                final_loss.backward()
                self.optim_schedule.step_and_update_lr()
            else:
                self.optim.zero_grad()
                final_loss.backward()
                self.optim.step()

            global_step = epoch * max_num_batches + i
            if i + 1 == max_num_batches:
                msg0 = 'epoch {:d}, '.format(epoch)
                msg1 = ""
                for k, v in cuml_loss_dict.items():
                    msg1 += "avg {}: {:.3f}, ".format(k, v / max_num_batches)
                msg2 = " . . .  avg loss: {:.5f}, avg reg loss: {:.5f}"
                msg2 = msg2.format(
                    sum(cuml_loss_dict.values()) / max_num_batches / len(self.experiment_names),
                    cuml_reg_loss / max_num_batches / len(self.train_config.regularization))
                desc2 = msg0 + msg1 + msg2
                pbar.set_description(desc2)

            if (global_step + 1) % self.train_config.log_freq == 0:
                self.writer.add_scalar("tot_loss", final_loss.item(), global_step)
                # TODO: add writer to print stuff during training
                # self.writer.add_embedding(
                #   self.model.get_word_embeddings(self.device),
                #   metadata=list(self.model.nlp.i2w.values()),
                #   global_step=global_step,
                #   tag='word_emb')

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

    def evaluate_model(self):
        train_preds_dict = self.generante_prediction(mode='train')
        valid_preds_dict = self.generante_prediction(mode='valid')

        train_nnll, valid_nnll = {}, {}
        train_r2, valid_r2 = {}, {}

        for expt, (true, pred) in train_preds_dict.items():
            train_nnll.update({expt: get_null_adj_nll(true, pred)})
            train_r2.update({expt: r2_score(true, pred, multioutput='raw_values') * 100})

        for expt, (true, pred) in valid_preds_dict.items():
            valid_nnll.update({expt: get_null_adj_nll(true, pred)})
            valid_r2.update({expt: r2_score(true, pred, multioutput='raw_values') * 100})

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
        # TODO: does this need to be updated for DataParallel?
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
                hidden_size=512,
                n_warmup_steps=self.train_config.warmup_steps,
            )

        else:
            raise ValueError("Invalid optimizer choice: {}".format(self.train_config.optim_chioce))

    def _setup_data(self):
        dataset = create_datasets(self.config, self.train_config.xv_folds, self.rng)
        self.dataset = dataset

        self.train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.train_config.batch_size,
            drop_last=True,
        )


def _send_to_cuda(data_tuple, device, dtype=torch.float32):
    return tuple(map(lambda z: z.to(device=device, dtype=dtype, non_blocking=False), data_tuple))
