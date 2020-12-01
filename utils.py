import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs/")


def CUDA(variable):
    """
    Put tensor `variable` to cuda device.
    """
    return variable.cuda() if torch.cuda.is_available() else variable


def CPU(variable):
    """
    Put tensor `variable` to cpu device.
    """
    return variable.cpu().detach()


def calc_accuracy(model, data_loader):
    """
    Calc model accuracy in specific data loader.
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = CUDA(images)
            labels = CUDA(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def workers_stat(workers):
    """
    compute the proportion of maclicious workers in the current iteration
    """
    total = len(workers)
    benign = np.sum(list(map(lambda w: int(w.role), workers)))
    return float(benign) / total


def build_net(features_dim, output_dim, hidden_sizes, activation, output_activation):
    sizes = [features_dim] + hidden_sizes + [output_dim]
    layers = []
    for i in range(len(sizes) - 1):
        layer = nn.Linear(sizes[i], sizes[i + 1])
        active = activation if i < len(hidden_sizes) - 1 else output_activation
        layers += [layer, active]
    return nn.Sequential(*layers)


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)
    return torch.cat((x1, x2), 1)


def collect_grads(model, loss):
    model.zero_grad()
    # with this line invoked, the gradient has been computed
    loss.backward()
    grads = []
    # # collect the gradients
    with torch.no_grad():
        _queue = [model]
        while len(_queue) > 0:
            cur = _queue[0]
            _queue = _queue[1:]  # dequeue
            if "weight" in cur._parameters:
                grads.append(
                    cur._parameters["weight"].grad.data.clone().view(-1).unsqueeze(-1)
                )
            if "bias" in cur._parameters and not (cur._parameters["bias"] is None):
                grads.append(
                    cur._parameters["bias"].grad.data.clone().view(-1).unsqueeze(-1)
                )
            for module in cur.children():
                _queue.append(module)
        # do the concantenate here
        grads = torch.cat(grads)
    return grads
