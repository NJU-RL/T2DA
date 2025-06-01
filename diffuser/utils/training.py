import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        save_freq=1000,
        bucket=None,
        train_device='cuda',
        is_evaluate: bool = False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.optimizer = torch.optim.Adam(
            diffusion_model.parameters(), lr=train_lr)

        self.bucket = bucket

        self.reset_parameters()
        self.step = 0
        if not is_evaluate:
            self.writer = SummaryWriter(f'{bucket}/tensorboard/train')

        self.device = train_device

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()

        '''
        min_value, max_value = np.infty, -np.infty
        for step in tqdm(range(10000)):
            trajectories, conditions, returns = next(self.dataloader)
            if min(returns) < min_value:
                min_value = min(returns)
            if max(returns) > max_value:
                max_value = max(returns)
        print(f'return scale: [{min_value}, {max_value}]')
        assert True==False
        '''

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                # batch: trajectories, conditions, returns, prompts
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                loss, infos = self.model.loss(*batch)

                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step += 1

            self.writer.add_scalar('diffusion/train loss', loss, self.step)
            self.writer.add_scalar('diffusion/train a0 loss', infos['a0_loss'], self.step)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join(
                    [f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(
                    f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k: v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.get_state_dict(),
            # 'ema': self.ema_model.get_state_dict()
        }
        savepath = os.path.join(self.bucket, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        savepath = os.path.join(savepath, f'state_{self.step}.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def load(self):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        # self.ema_model.load_state_dict(data['ema'], strict=False)
        self.ema_model.load_state_dict(data['model'], strict=False)