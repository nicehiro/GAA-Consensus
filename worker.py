from enum import Enum
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from adv_loss import adv_losses
from models.cifar10 import Cifar10
from models.mnist import Mnist
from models.policy import Policy
from utils import CUDA, collect_grads, preprocess_gradients


class Role(Enum):
    NORMAL = 0
    TRADITIONAL_ATTACK = 1
    ADVERSERY_ATTACK = 2
    MISSING_LABEL = 3


class Worker:
    meta_models = {"MNIST": Mnist, "CIFAR10": Cifar10}

    def __init__(
        self,
        wid,
        atk_fn,
        adv_loss,
        neighbors_n,
        train_loader,
        test_loader,
        meta_lr=1e-2,
        policy_lr=1e-2,
        dataset="MNIST",
        missing_labels=None,
        role=Role.NORMAL,
        period=2e8,
        alpha=0.5,
        extreme_mail=None,
        pretense=1e8,
    ) -> None:
        self.wid = wid
        self.atk_fn = atk_fn
        self.role = role
        self.period = period
        self.neighbors_n = neighbors_n
        self.alpha = Variable(
            torch.tensor([1.0 / neighbors_n for _ in range(neighbors_n)]),
            requires_grad=True,
        )
        self.alpha = CUDA(self.alpha)
        self.dim = 0
        self.adv_loss = adv_loss
        self.missing_labels = missing_labels
        self.extreme_mail = extreme_mail
        self.pretense = pretense
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_x, self.val_y = self._generate_val_set()
        self.meta_model = CUDA(self.meta_models[dataset]())
        self.meta_model_copy = CUDA(self.meta_models[dataset]())
        self.meta_lr = meta_lr
        self.policy_lr = policy_lr
        # TODO remove hard code
        self.policy = CUDA(Policy(32, 10, 0, 0))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.criterion = F.cross_entropy
        self.rollback_buffer = []

    def reset(self):
        """
        When calc meta loss, will generate grads in params, but not be used.
        So when calc policy loss, will get error.
        """
        # self.reset_meta_model()
        self.copy_meta_params_to(self.meta_model_copy)
        self.copy_meta_params_from(self.meta_model_copy)
        self.alpha = CUDA(Variable(self.alpha.data))

    def reset_meta_model(self):
        """
        Reset meta model's grads.
        """
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                cur._parameters["weight"] = Variable(cur._parameters["weight"].data)
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                cur._parameters["bias"] = Variable(cur._parameters["bias"].data)
            for module in cur.children():
                _queue.append(module)

    def _generate_val_set(self):
        val_x, val_y = self.train_loader.val_set()
        val_x, val_y = CUDA(val_x), CUDA(val_y)
        return val_x, val_y

    def submit(self, iter_no=0):
        x, y = self.train_loader.next()
        x, y = Variable(x), Variable(y)
        x, y = CUDA(x), CUDA(y)
        if self.role is Role.NORMAL:
            grads = self._normal_submit(x, y)
        elif self.role is Role.TRADITIONAL_ATTACK:
            grads = self._traditional_submit(x, y)
        elif self.role is Role.ADVERSERY_ATTACK:
            grads = self._adversary_submit(x, y)
        elif self.role is Role.MISSING_LABEL:
            grads = self._missing_label_submit(x, y, iter_no=0)
        else:
            raise Exception("No such a Role: {0}".format(self.role))
        return grads.detach()

    def _normal_submit(self, x, y):
        predict_y = self.meta_model(x)
        loss = self.criterion(predict_y, y)
        return collect_grads(self.meta_model, loss)

    def _traditional_submit(self, x, y):
        grads = self._normal_submit(x, y)
        return self.atk_fn(grads, sigma=2e-6)

    def _adversary_submit(self, x, y):
        return self._traditional_submit(x, y)

    def _missing_label_submit(self, x, y, iter_no):
        def divide_data(x, y, missing_labels):
            x, y = x.numpy(), y.numpy()
            mask = np.isin(y, missing_labels)
            x1, y1 = torch.from_numpy(x[~mask]), torch.from_numpy(y[~mask])
            x2, y2 = torch.from_numpy(x[mask]), torch.from_numpy(y[mask])
            return x1, y1, x2, y2

        x, y, x1, y1 = divide_data(x, y, self.missing_labels)
        # used labels in validation set
        x, y = x.cuda(), y.cuda()
        f_x = self.meta_model(x)
        loss2 = self.criterion(f_x, y)
        # other data
        loss1 = 0.0
        if x1.shape[0] > 0:
            x1, y1 = x1.cuda(), y1.cuda()
            val_fx = self.meta_model(x1)
            loss1 = self.criterion(val_fx, y1)
        grads = adv_losses[self.adv_loss](self.meta_model, loss1, loss2, iter_no)
        return grads

    def meta_update(self, Q):
        """
        Only for server update meta model.

        Gather all workers' gradient then update.
        Calc loss of old meta model.
        """
        flat_params = self.get_meta_model_flat_params().unsqueeze(-1)
        # update meta network using linear GAR
        for i in range(len(Q)):
            flat_params = flat_params - self.meta_lr * (
                CUDA(self.alpha[i]) * CUDA(Q[i])
            )
        self.set_meta_model_flat_params(flat_params)

    def policy_update(self, loss):
        """
        Update policy net.

        `loss` is the difference between curr policy loss and last policy loss.
        """
        torch.autograd.set_detect_anomaly(True)
        self.optimizer.zero_grad()
        loss.backward()
        for params in self.policy.parameters():
            params.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def alpha_update(self, Q):
        """
        Update alpha.
        """
        flat_params = self.get_meta_model_flat_params().unsqueeze(-1)
        self.alpha = self.alpha.expand(flat_params.size(0), -1)
        # calc meta loss on validation set
        predict_y = self.meta_model(self.val_x)
        loss = self.criterion(predict_y, self.val_y)
        loss_ = loss.detach().clone().data
        loss_ = loss.expand_as(flat_params)
        # calc policy action
        inputs = torch.cat((*map(preprocess_gradients, Q), flat_params.data, loss_), 1)
        inputs = torch.cat((inputs, self.alpha), 1)
        self.alpha = F.softmax(self.policy(inputs).mean(dim=0), dim=0)
        return loss

    def copy_meta_params_from(self, model):
        for modelA, modelB in zip(self.meta_model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_meta_params_to(self, model):
        for modelA, modelB in zip(self.meta_model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)

    def get_meta_model_flat_params(self):
        """
        Get all meta_model parameters.
        """
        params = []
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                params.append(cur._parameters["weight"].view(-1))
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                params.append(cur._parameters["bias"].view(-1))
            for module in cur.children():
                _queue.append(module)
        return torch.cat(params)

    def set_meta_model_flat_params(self, flat_params):
        """
        Restore original shapes (which is actually required during the training phase)
        """
        offset = 0
        _queue = [self.meta_model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            weight_flat_size = 0
            bias_flat_size = 0
            if "weight" in cur._parameters:
                weight_shape = cur._parameters["weight"].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                cur._parameters["weight"].data = flat_params[
                    offset : offset + weight_flat_size
                ].view(*weight_shape)
                # cur._parameters["weight"].grad = torch.zeros(*weight_shape)
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                bias_shape = cur._parameters["bias"].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                cur._parameters["bias"].data = flat_params[
                    offset
                    + weight_flat_size : offset
                    + weight_flat_size
                    + bias_flat_size
                ].view(*bias_shape)
                # cur._parameters["bias"].grad = torch.zeros(*bias_shape)
            offset += weight_flat_size + bias_flat_size
            for module in cur.children():
                _queue.append(module)
