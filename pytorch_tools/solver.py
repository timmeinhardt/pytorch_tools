"""Neural Network Training Solver."""
import time
import copy
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable


class Solver(object):
    """Neural Network Training Solver."""

    def __init__(self, optim, optim_kwargs, ada_lr, early_stopping, epochs,
                 loss, loss_kwargs=None, logger=None, train_log_interval=None):
        self._optim_kwargs = optim_kwargs
        self._ada_lr = ada_lr
        self._early_stopping = early_stopping
        self._epochs = epochs
        self._train_log_interval = train_log_interval

        self._optim_func = optim
        if isinstance(optim, str):
            self._optim_func = getattr(torch.optim, optim)

        self.loss_func = loss
        if isinstance(loss, str):
            self.loss_func = getattr(nn, loss)(**loss_kwargs)

        self._logger = logger
        self.reset()


    @property
    def last_metrics(self):
        """
        """
        train_loss = train_acc = val_loss = val_acc = None

        if self.train_hist['loss']:
            train_loss = self.train_hist['loss'][-1]
        if self.train_hist['acc']:
            train_acc = self.train_hist['acc'][-1]
        if self.val_hist['loss']:
            val_loss = self.val_hist['loss'][-1]
        if self.val_hist['acc']:
            val_acc = self.val_hist['acc'][-1]
        return train_loss, train_acc, val_loss, val_acc

    def best_metrics(self, n=1):
        """
        """
        train_loss = train_acc = val_loss = val_acc = None

        if self.train_hist['loss']:
            train_loss, _ = torch.Tensor(self.train_hist['loss']).sort(dim=0)
        if self.train_hist['acc']:
            train_acc, _ = torch.Tensor(self.train_hist['acc']).sort(dim=0, descending=True)
        if self.val_hist['loss']:
            val_loss, _ = torch.Tensor(self.val_hist['loss']).sort(dim=0)
        if self.val_hist['acc']:
            val_acc, _ = torch.Tensor(self.val_hist['acc']).sort(dim=0, descending=True)
        return train_loss[:n], train_acc[:n], val_loss[:n], val_acc[:n]

    @property
    def trained_epochs(self):
        """
        :returns: number of trained epochs
        :rtype: int
        """
        return len(self.train_hist['acc'])

    def early_stopping(self):
        """
        :returns: if solver should stop early
        :rtype: bool
        """
        patience = self._early_stopping['patience']
        min_delta = self._early_stopping['min_delta']
        metric = self._early_stopping['metric']

        # early stoppping deactivated
        if patience is None:
            return False

        _, _, best_val_loss, best_val_acc = self.best_metrics()

        for m in self.val_hist[metric][-patience:]:
            if metric == 'acc':
                if (m - best_val_acc).cpu().numpy() >= - 1.0 * min_delta:
                    return False
            else:
                if (m - best_val_loss).item() <= 1.0 * min_delta:
                    return False

        return True

    def reset(self):
        """Reset solver."""
        self.best_val_model = self.optim = None
        self.train_hist = dict(loss=[], acc=[])
        self.val_hist = dict(loss=[], acc=[])

    def train(self, model, data_loader, epochs=None, save_hist=True,
              vis_callback=None):
        """Train model specified epochs on data_loader."""
        model.train()
        device = model.device

        for _ in self.epochs_iter(epochs):
            if self.trained_epochs in self._ada_lr['epochs']:
                # TODO: make optim global and include Scheduler
                for param_group in self.optim.param_groups:
                    param_group['lr'] *= self._ada_lr['factor']

            loss, acc_sum = 0.0, 0
            for batch_id, (data, target) in enumerate(data_loader, 1):
                data, target = data.to(device), target.to(device)

                self.optim.zero_grad()
                output = model(data)
                batch_loss = self.loss_func(output, target)
                batch_loss.backward()
                self.optim.step()

                batch_acc = self.compute_acc(output, target, data)
                loss += batch_loss.item()
                acc_sum += batch_acc.item()

                # logging
                if (self._train_log_interval is not None
                        and not batch_id % self._train_log_interval):
                    assert self._logger is not None
                    max_epochs = len(self.epochs_iter(self._epochs))
                    log_msg = (
                        f"TRAIN = [Epoch {self.trained_epochs}/{max_epochs}] "
                        f"[Samples {batch_id * len(data)}/"
                        f"{data_loader.num_samples}] "
                        f"Batch {batch_id} "
                        f"loss/acc {batch_loss.item() / len(data):.4f}/"
                        f"{batch_acc.item() / len(data):.2%} "
                        f"({batch_acc.item()}/{len(data)})")
                    self._log(log_msg)

                if vis_callback is not None:
                    vis_callback(self, batch_loss, batch_acc, data, target, output, batch_id)

            loss /= data_loader.num_samples
            acc = acc_sum / data_loader.num_samples
            if save_hist:
                self.train_hist['acc'].append(acc)
                self.train_hist['loss'].append(loss)

        return loss, acc

    def test(self, data_loader, model=None):
        """Test best val model on data_loader."""
        if model is None:
            assert self.best_val_model is not None, (
                "The best validation model is None and can not be tested."
                "Please provide a model file or save model during training.")
            model = self.best_val_model

        loss, acc = self._infer_data_loader(model, data_loader)
        self._log(f"TEST = loss/acc: {loss:.4f}/{acc:.2%}")

        return loss, acc

    def train_val(self, model, train_loader, val_loader, epochs=None,
                  vis_callback=None, train_vis_callback=None,
                  save_best_model=True, reinfer_train_loader=False):
        """Train and validate after each epoch."""
        for epoch in self.epochs_iter(epochs):
            start = time.time()
            # train for one epoch
            train_loss, train_acc = self.train(model, train_loader, epochs=1,
                vis_callback=train_vis_callback, save_hist=False)

            # infer train and val metrics
            # if trained with for example dropout one has to reinfer to obtain
            # meaningful train metrics
            if reinfer_train_loader:
                train_loss, train_acc = self._infer_data_loader(model, train_loader)
            val_loss, val_acc = self._infer_data_loader(model, val_loader)
            self.train_hist['loss'].append(train_loss)
            self.train_hist['acc'].append(train_acc)
            self.val_hist['loss'].append(val_loss)
            self.val_hist['acc'].append(val_acc)
            _, _, _, best_val_acc = self.best_metrics()

            if save_best_model and (val_acc == best_val_acc).item():
                self.best_val_model = copy.deepcopy(model)

            if vis_callback is not None:
                vis_callback(self, epoch, time.time() - start)

            if self.early_stopping():
                break

        # logging
        self._log(f"TRAIN = loss/acc: {train_loss:.4f}/{train_acc:.2%}")
        self._log(f"VAL   = loss/acc: {val_loss:.4f}/{val_acc:.2%}")

        return train_loss, train_acc, val_loss, val_acc

    def _log(self, log_msg):
        if self._logger is not None:
            self._logger(log_msg)

    def _infer_data_loader(self, model, data_loader):
        """Infer entire data loader and evaluate model."""
        model.eval()
        device = model.device

        loss, acc_sum = 0.0, 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)
                loss += self.loss_func(output, target).item()
                acc_sum += self.compute_acc(output, target, data).item()

        model.train()

        loss /= data_loader.num_samples
        acc = acc_sum / data_loader.num_samples

        return loss, acc

    def epochs_iter(self, epochs):
        """Epochs iterator with support for infinite iterations."""
        if epochs is None:
            epochs = self._epochs

        if epochs is None:
            epochs_iter = itertools.count()
        else:
            epochs_iter = range(1, epochs + 1)
        return epochs_iter

    def init_optim(self, model):
        self.optim = self._optim_func(model.parameters(), **self._optim_kwargs)

    @staticmethod
    def compute_acc(output, target, data):
        """
        : returns: number of correct predictions
        : rtype: torch.IntTensor of size 1
        """
        _, pred = output.data.max(dim=1)
        return pred.eq(target.data).sum(dim=0).int()

