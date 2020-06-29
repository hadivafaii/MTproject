import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
from datetime import datetime
from sklearn.metrics import r2_score

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from .dataset import create_datasets
from .optimizer import Lamb, log_lamb_rs, ScheduledOptim

import sys; sys.path.append('..')
from utils.generic_utils import to_np, compute_reg_loss, get_null_adj_nll


class MTTrainer:
    def __init__(self, model, train_config, seed=665):

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)

        cuda_condition = torch.cuda.is_available() and train_config.use_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.model = model.to(self.device)
        self.model.reg_dicts_to_device(self.device)
        self.train_config = train_config
        self.config = model.config

        self.datasets_dict = create_datasets(self.config)
        self.train_valid_indxs = {}
        self.train_loaders_dict = {}
        self.valid_loaders_dict = {}
        self._create_train_valid_loaders()

        self.writer = None

        self.optim = None
        self.optim_schedule = None
        self.setup_optim()

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
            pjoin(self.train_config.runs_dir, "{}_{}".format(comment, datetime.now().strftime("[%Y_%m_%d_%H:%M]"))))

        self.model.train()

        train_preds_dict, valid_preds_dict = {}, {}
        epochs_range = range(nb_epochs) if isinstance(nb_epochs, int) else nb_epochs
        for epoch in epochs_range:
            self.iteration(self.train_loaders_dict, epoch=epoch)

            if (epoch + 1) % self.train_config.chkpt_freq == 0:
                print('Saving chkpt:{:d}'.format(epoch+1))
                train_preds_dict, valid_preds_dict = self.evaluate_model()
                self.model.save('chkpt:{:d}'.format(epoch+1), comment=comment)

        return train_preds_dict, valid_preds_dict

    def evaluate_model(self):
        train_preds_dict = self.generante_prediction(mode='train')
        valid_preds_dict = self.generante_prediction(mode='valid')

        nnll_all = []
        r2_all = []
        for expt, (true, pred) in train_preds_dict.items():
            nnll_all.append(get_null_adj_nll(true, pred))
            r2_all.append(r2_score(true, pred, multioutput='raw_values') * 100)

        print('avg train nnll: {:.3f}, r2: {:.2f} {}'.format(np.mean(nnll_all), np.mean(r2_all), '%'))

        nnll_all = []
        r2_all = []
        for expt, (true, pred) in valid_preds_dict.items():
            nnll_all.append(get_null_adj_nll(true, pred))
            r2_all.append(r2_score(true, pred, multioutput='raw_values') * 100)

        print('avg valid nnll: {:.3f}, r2: {:.2f} {}'.format(np.mean(nnll_all), np.mean(r2_all), '%'))

        return train_preds_dict, valid_preds_dict

    def iteration(self, dataloaders_dict, epoch=0):
        cuml_loss = 0.0
        cuml_reg_loss = 0.0

        data_iterators = {k: iter(v) for k, v in dataloaders_dict.items()}
        max_num_batches = max([len(dataloader) for dataloader in dataloaders_dict.values()])

        pbar = tqdm(range(max_num_batches))
        for i in pbar:
            losses_dict = {}
            for expt, iterator in data_iterators.items():
                try:
                    data_tuple = next(iterator)
                except StopIteration:
                    data_iterators[expt] = iter(dataloaders_dict[expt])
                    data_tuple = next(data_iterators[expt])

                batch_data_tuple = _send_to_cuda(data_tuple, self.device)

                batch_inputs = batch_data_tuple[0]
                batch_targets = batch_data_tuple[1]

                pred = self.model(batch_inputs)
                loss = self.model.criterion(pred, batch_targets)
                losses_dict.update({expt: loss})
                cuml_loss += loss.item()

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

            total_loss = sum(x for x in losses_dict.values()) / len(losses_dict)
            total_reg_loss = sum(x for x in reg_losses_dict.values()) / len(reg_losses_dict)
            cuml_reg_loss += total_reg_loss.item()

            final_loss = total_loss + total_reg_loss

            # backward and optimization only in train
            if self.train_config.optim_choice == 'adam_with_warmup':
                self.optim_schedule.zero_grad()
                final_loss.backward()
                self.optim_schedule.step_and_update_lr()
            else:
                self.optim.zero_grad()
                final_loss.backward()
                self.optim.step()

            msg0 = 'epoch {:d}'.format(epoch)
            msg1 = ""
            for k, v in losses_dict.items():
                msg1 += "{}: {:.3f}, ".format(k, v.item())
            msg1 += "tot: {:.3f}, ".format(total_loss.item())
            msg1 += "tot reg: {:.2e}".format(total_reg_loss.item())

            desc1 = msg0 + '\t|\t' + msg1
            pbar.set_description(desc1)

            global_step = epoch * max_num_batches + i
            if i + 1 == max_num_batches:
                desc2 = 'epoch # {:d}, avg loss: {:.5f}, avg reg loss: {:.5f}'
                desc2 = desc2.format(
                    epoch,
                    cuml_loss / max_num_batches / len(dataloaders_dict),
                    cuml_reg_loss / max_num_batches / len(self.train_config.regularization))
                pbar.set_description(desc2)

                # self.writer.add_embedding(
                #   self.model.get_word_embeddings(self.device),
                #   metadata=list(self.model.nlp.i2w.values()),
                #   global_step=global_step,
                #   tag='word_emb')

    def generante_prediction(self, mode='train'):
        self.model.eval()
        preds_dict = {}

        if mode == 'train':
            for expt, loader in self.train_loaders_dict.items():
                true_list = []
                pred_list = []

                for i, data_tuple in enumerate(loader):
                    with torch.no_grad():
                        pred = self.model(data_tuple[0].to(self.device, dtype=torch.float))

                    true_list.append(data_tuple[1])
                    pred_list.append(pred)

                true = torch.cat(true_list)
                pred = torch.cat(pred_list)

                preds_dict.update({expt: (to_np(true), to_np(pred))})

        elif mode == 'valid':
            for expt, loader in self.valid_loaders_dict.items():
                true_list = []
                pred_list = []

                for i, data_tuple in enumerate(loader):
                    with torch.no_grad():
                        pred = self.model(data_tuple[0].to(self.device, dtype=torch.float))

                    true_list.append(data_tuple[1])
                    pred_list.append(pred)

                true = torch.cat(true_list)
                pred = torch.cat(pred_list)

                preds_dict.update({expt: (to_np(true), to_np(pred))})
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        return preds_dict

    def setup_optim(self):
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
                hidden_size=self.model.config.grid_size ** 2,
                n_warmup_steps=self.train_config.warmup_steps,
            )

            # print('\nUsing {} with {} warmup steps.  Large vs small lr doesnt apply. . . '.format(
            #     self.train_config.optim_choice, self.train_config.warmup_steps))

        else:
            raise ValueError("Invalid optimizer choice: {}".format(self.train_config.optim_chioce))

    def _create_train_valid_loaders(self):
        for expt, dataset in self.datasets_dict.items():
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            self.rng.shuffle(indices)
            split = int(np.floor(self.train_config.validation_split * dataset_size))

            trn_indices, val_indices = indices[split:], indices[:split]
            train_loader = DataLoader(
                dataset=dataset,
                sampler=SubsetRandomSampler(trn_indices),
                batch_size=self.train_config.batch_size,
                pin_memory=True,
            )
            valid_loader = DataLoader(
                dataset=dataset,
                sampler=SubsetRandomSampler(val_indices),
                batch_size=self.train_config.batch_size,
                pin_memory=True,
            )

            self.train_valid_indxs.update({expt: (trn_indices, val_indices)})
            self.train_loaders_dict.update({expt: train_loader})
            self.valid_loaders_dict.update({expt: valid_loader})


def _send_to_cuda(data_tuple, device, dtype=torch.float32):
    return tuple(map(lambda z: z.to(device=device, dtype=dtype, non_blocking=True), data_tuple))
