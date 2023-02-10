import numpy as np
import torch
from torchvision.utils import make_grid
from datetime import datetime, timedelta
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, models, criterion, metric_ftns, optimizers, config, device,
                 data_loader, scalers, amp_autocast, valid_data_loader=None, lr_schedulers=None, len_epoch=None):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.scalers = scalers
        self.amp_autocast = amp_autocast
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_schedulers
        # self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = 50

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.weight_penalties = [100.0, 1.0, 1000000.0]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for i in range(3):
            self.models[i].train()
        self.train_metrics.reset()

        start_time = datetime.now()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data = data.to(self.device)

            t = [0, 0, 0]
            t[0] = target[:, [0, 3]].to(self.device)
            t[1] = target[:, [1, 2]].to(self.device)
            t[2] = target[:, [4, 5]].to(self.device)

            # self.optimizer.zero_grad()
            # output = self.model(data)
            # loss = self.criterion(output, target)
            # loss.backward()
            # self.optimizer.step()

            overall_predictions = []
            batch_loss = 0.0
            for i in range(3):
                with self.amp_autocast():
                    output = self.models[i](data)
                    loss = self.criterion(output, t[i], self.weight_penalties[i], self.device)

                self.scalers[i].scale(loss).backward()
                self.scalers[i].step(self.optimizers[i])
                self.scalers[i].update()
                self.optimizers[i].zero_grad()

                batch_loss += loss.item()
                overall_predictions.append(output[:, 0].detach().cpu().reshape(-1, 1))
                overall_predictions.append(output[:, 1].detach().cpu().reshape(-1, 1))

            overall_prediction = torch.cat([overall_predictions[0],
                                            overall_predictions[2],
                                            overall_predictions[3],
                                            overall_predictions[1],
                                            overall_predictions[4],
                                            overall_predictions[5]], dim=1)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', batch_loss)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(overall_prediction, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    batch_loss))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        elapsed_time_secs = (datetime.now() - start_time).seconds

        val_start_time = datetime.now()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        val_elapsed_time_secs = (datetime.now() - val_start_time).seconds

        additional_log = {'Epoch took': timedelta(seconds=elapsed_time_secs),
                          'Validation took': timedelta(seconds=val_elapsed_time_secs)}
        log.update(additional_log)

        # if self.lr_scheduler:
        #     self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        for i in range(3):
            self.models[i].eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                t = [0, 0, 0]
                t[0] = target[:, [0, 3]].to(self.device)
                t[1] = target[:, [1, 2]].to(self.device)
                t[2] = target[:, [4, 5]].to(self.device)

                overall_prediction = []
                batch_loss = 0.0
                for i in range(3):
                    output = self.models[i](data)
                    loss = self.criterion(output, t[i], self.weight_penalties[i], self.device)

                    batch_loss += loss.item()
                    overall_prediction.append(output[:, 0].detach().cpu().reshape(-1, 1))
                    overall_prediction.append(output[:, 1].detach().cpu().reshape(-1, 1))

                overall_prediction = torch.cat([overall_prediction[0],
                                                overall_prediction[2],
                                                overall_prediction[3],
                                                overall_prediction[1],
                                                overall_prediction[4],
                                                overall_prediction[5]], dim=1)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', batch_loss)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(overall_prediction, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
