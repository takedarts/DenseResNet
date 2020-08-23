import torch
import torch.nn as nn
import torch.autograd as autograd
import apex

import logging

from .cutmix import cutmix
from .mixup import mixup

LOGGER = logging.getLogger(__name__)


class Trainer(object):

    @staticmethod
    def optimizer2devices(optimizer, devices):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(devices)

    def __init__(self, model, optimizer, devices=None, opt=0):
        super().__init__()

        if devices is not None and len(devices) != 0:
            model = model.cuda(devices[0])
            Trainer.optimizer2devices(optimizer, devices[0])

            if opt != 0:
                model, optimizer = apex.amp.initialize(
                    model, optimizer, opt_level='O{}'.format(opt),
                    verbosity=int(LOGGER.isEnabledFor(logging.DEBUG)))

            if len(devices) > 1:
                model = nn.DataParallel(model, devices)

        self.model = model
        self.optimizer = optimizer
        self.devices = devices
        self.opt = opt
        self.reset()

    def reset(self):
        self._loss_value = 0
        self._loss_count = 0
        self._accuracy1_value = 0
        self._accuracy5_value = 0
        self._accuracy_count = 0

    @property
    def loss(self):
        return self._loss_value / max(self._loss_count, 1)

    @property
    def accuracy1(self):
        return self._accuracy1_value / max(self._accuracy_count, 1)

    @property
    def accuracy5(self):
        return self._accuracy5_value / max(self._accuracy_count, 1)

    def get_status(self):
        return {'loss': self.loss, 'accuracy1': self.accuracy1, 'accuracy5': self.accuracy5}

    def _update_model(self, loss):
        self.optimizer.zero_grad()

        if self.opt == 0:
            loss.backward()
        else:
            with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

        self.optimizer.step()

    def _update_loss(self, preds, targets):
        loss = (-1 * nn.functional.log_softmax(preds, dim=1) * targets).sum(dim=1).mean()
        self._loss_value += float(loss.detach().cpu()) * preds.shape[0]
        self._loss_count += preds.shape[0]

        return loss

    def _update_accuracy(self, preds, targets):
        corrects = (preds.topk(5, dim=1)[1] == targets.max(dim=1)[1].view(-1, 1)).float()
        self._accuracy1_value += float(corrects[:, 0].sum())
        self._accuracy5_value += float(corrects.max(dim=1)[0].sum())
        self._accuracy_count += preds.shape[0]

    def train(self, loader, label_smooth=0,
              cutmix_prob=0, cutmix_beta=1, mixup_prob=0, mixup_beta=1):
        self.reset()
        self.model.train()

        for images, targets in loader:
            if len(self.devices) != 0:
                images = images.cuda(self.devices[0], non_blocking=True)
                targets = targets.cuda(self.devices[0], non_blocking=True)

            if label_smooth != 0:
                targets *= 1 - label_smooth
                targets += label_smooth / targets.shape[1]

            if cutmix_prob != 0 and cutmix_beta != 0:
                images, targets = cutmix(images, targets, cutmix_prob, cutmix_beta)

            if mixup_prob != 0 and mixup_beta != 0:
                images, targets = mixup(images, targets, mixup_prob, mixup_beta)

            if LOGGER.isEnabledFor(logging.DEBUG):
                with autograd.detect_anomaly():
                    preds = self.model(images)
            else:
                preds = self.model(images)

            self._update_model(self._update_loss(preds, targets))
            self._update_accuracy(preds, targets)

            LOGGER.debug('train: %s', self)

    def tune(self, loader):
        self.reset()
        self.model.eval()
        self.model.classifier.train()

        for images, targets in loader:
            if len(self.devices) != 0:
                images = images.cuda(self.devices[0], non_blocking=True)
                targets = targets.cuda(self.devices[0], non_blocking=True)

            with torch.no_grad():
                feats = self.model.get_features(images)[0].detach()

            if LOGGER.isEnabledFor(logging.DEBUG):
                with autograd.detect_anomaly():
                    preds = self.model.get_output(feats)
            else:
                preds = self.model.get_output(feats)

            self._update_model(self._update_loss(preds, targets))
            self._update_accuracy(preds, targets)

            LOGGER.debug('tune: %s', self)

    def validate(self, loader):
        self.reset()
        self.model.eval()

        for images, targets in loader:
            if len(self.devices) != 0:
                images = images.cuda(self.devices[0], non_blocking=True)
                targets = targets.cuda(self.devices[0], non_blocking=True)

            with torch.no_grad():
                preds = self.model(images)

            self._update_loss(preds, targets)
            self._update_accuracy(preds, targets)

            LOGGER.debug('valid: %s', self)

    def __str__(self):
        return (
            f'loss={self.loss:.4f}'
            f', accuracy1={self.accuracy1:.4f}'
            f', accuracy5={self.accuracy5:.4f}')
