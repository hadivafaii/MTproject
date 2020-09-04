import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model_utils import get_null_adj_nll, print_num_params
import sys; sys.path.append('..')
from utils.generic_utils import to_np


class SingleCellReadout(nn.Module):
    def __init__(self,
                 model_dim,
                 nb_sk,
                 nb_tk,
                 lr: float = 1e-3,
                 wd: float = 2.0,
                 tmax: int = 10,
                 eta_min: float = 1e-6,
                 verbose=False,):
        super(SingleCellReadout, self).__init__()

        self.temporal_fcs = nn.ModuleList([
            nn.Linear(11, nb_tk[0], bias=False),
            nn.Linear(6, nb_tk[1], bias=False),
            nn.Linear(3, nb_tk[2], bias=False),
        ])
        self.spatial_fcs = nn.ModuleList([
            nn.Linear(225, nb_sk[0], bias=False),
            nn.Linear(64, nb_sk[1], bias=False),
            nn.Linear(16, nb_sk[2], bias=False),
        ])

        self.relu = nn.ReLU(inplace=True)
        model_dims = [model_dim, model_dim * 2, model_dim * 4]
        num_filters = sum([np.prod(item) for item in zip(nb_tk, nb_sk, model_dims)])
        self.layer = nn.Linear(num_filters, 1)
        self.softplus = nn.Softplus()
        self.criterion = nn.PoissonNLLLoss(log_input=False, reduction="sum")

        self.optim = None
        self.optim_schedule = None
        self._setup_optim(lr, wd, tmax, eta_min)

        if verbose:
            print_num_params(self)

    def forward(self, x1, x2, x3):
        x1 = self.temporal_fcs[0](x1).permute(0, 1, -1, 2)
        x2 = self.temporal_fcs[1](x2).permute(0, 1, -1, 2)
        x3 = self.temporal_fcs[2](x3).permute(0, 1, -1, 2)

        z1 = self.spatial_fcs[0](x1).flatten(start_dim=1)
        z2 = self.spatial_fcs[1](x2).flatten(start_dim=1)
        z3 = self.spatial_fcs[2](x3).flatten(start_dim=1)

        z = torch.cat([z1, z2, z3], dim=-1)
        z = self.relu(z)

        y = self.layer(z)
        y = self.softplus(y)

        return y

    def trn(self, trainer, epoch=0, cell=None):
        self.train()

        max_num_batches = len(trainer.readout_train_loader)
        pbar = tqdm(enumerate(trainer.readout_train_loader), total=max_num_batches)

        cuml_loss = 0.0
        for i, data in pbar:
            _, x1, x2, x3 = data[0]
            y = data[1]
            filt = data[2]

            if cell is not None:
                y = y[:, [cell]]
                filt = filt[:, [cell]]

            x1_pt = x1.float().cuda().flatten(start_dim=2, end_dim=3)
            x2_pt = x2.float().cuda().flatten(start_dim=2, end_dim=3)
            x3_pt = x3.float().cuda().flatten(start_dim=2, end_dim=3)
            y_pt = y.float().cuda()

            pred = self(x1_pt, x2_pt, x3_pt)

            if filt is not None:
                filt = filt.float().cuda()
                loss = self.criterion(pred * filt, y_pt * filt) / trainer.readout_train_loader.batch_size
            else:
                loss = self.criterion(pred, y_pt) / trainer.readout_train_loader.batch_size

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            msg = "epoch: {}, loss: {:.3f}".format(epoch, loss.item())
            pbar.set_description(msg)
            cuml_loss += loss.item()
            if (i + 1) == max_num_batches:
                msg = "epoch # {}. avg loss: {:.4f}".format(epoch, cuml_loss / max_num_batches)
                pbar.set_description(msg)

        self.optim_schedule.step()

    def tst(self, trainer, epoch=0, cell=None):
        self.eval()

        pred_list = []
        true_list = []
        for data in trainer.readout_valid_loader:
            _, x1, x2, x3 = data[0]
            y = data[1]
            filt = data[2]

            if cell is not None:
                y = y[:, [cell]]
                filt = filt[:, [cell]]

            x1_pt = x1.float().cuda().flatten(start_dim=2, end_dim=3)
            x2_pt = x2.float().cuda().flatten(start_dim=2, end_dim=3)
            x3_pt = x3.float().cuda().flatten(start_dim=2, end_dim=3)
            y_pt = y.float().cuda()

            with torch.no_grad():
                pred = self(x1_pt, x2_pt, x3_pt)
                if filt is not None:
                    filt = filt.float().cuda()
                    loss = self.criterion(pred * filt, y_pt * filt) / trainer.readout_train_loader.batch_size
                else:
                    loss = self.criterion(pred, y_pt) / trainer.readout_train_loader.batch_size

            pred_list.append(pred)
            true_list.append(y_pt)

        pred = torch.cat(pred_list)
        true = torch.cat(true_list)

        pred = to_np(pred)
        true = to_np(true)

        nnll = get_null_adj_nll(pred, true)

        msg = "====> epoch # {}. avg nnll: {:.3f},  meadian nnll: {:.3f},  test loss: {:.2f},\n"
        msg = msg.format(epoch, np.mean(nnll), np.median(nnll), loss)
        print(msg)

        return nnll, pred, true

    def _setup_optim(self, lr, wd, tmax, eta_min):
        self.optim = AdamW(self.parameters(), lr=lr, weight_decay=wd)
        self.optim_schedule = CosineAnnealingLR(self.optim, T_max=tmax, eta_min=eta_min)
