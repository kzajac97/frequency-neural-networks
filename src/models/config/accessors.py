import math

import torch
from losses import LogCoshLoss


# activation mapping dict allowing to access activation functions by name in sweep config
ACTIVATION_MAPPING = {
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
    "leaky_relu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "softmin": torch.nn.Softmin,
    "softmax": torch.nn.Softmax,
    "log_softmax": torch.nn.LogSoftmax,
}

LOSS_MAPPING = {
    "mse": torch.nn.MSELoss,
    "l1": torch.nn.L1Loss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "log_cosh": LogCoshLoss,
}


OPTIMIZER_MAPPING = {
    "adam": torch.optim.Adam,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
    "rmsprop": torch.optim.RMSprop,
    "sgd": torch.optim.SGD,
}
