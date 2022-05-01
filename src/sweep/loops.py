from typing import Callable
from typing import List

import numpy as np
import pandas as pd
import torch

from datasets import TorchSimulationSequenceLoader
from metrics import regression_score
from sweep.callbacks import AbstractCallback
from utils.tensors import detach_to_numpy


def evaluate(model: torch.nn.Module, generator: TorchSimulationSequenceLoader) -> pd.Series:
    """
    Evaluate the model on the test set

    :param model: the model to evaluate
    :param generator: the data loader to use

    :return: a pandas series containing the evaluation metrics
    """
    model.eval()
    predictions, targets = [], []

    for x, y in generator.test_loader:
        y_pred = model(x)
        y_true, y_pred = detach_to_numpy(y, y_pred)
        predictions.append(y_pred)
        targets.append(y_true)

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return regression_score(y_pred=predictions.flatten(), y_true=targets.flatten())


def train(
    model: torch.nn.Module,
    generator: TorchSimulationSequenceLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    callbacks: List[AbstractCallback] = None,
) -> torch.nn.Module:
    """
    Train the model.

    :param model: The model to train.
    :param generator: The data loader to use.
    :param loss: The loss function to use.
    :param optimizer: The optimizer to use.
    :param callbacks: A list of callbacks to use.
    :param n_epochs: The number of epochs to train for.

    :return: fitted model
    """

    for epoch in range(n_epochs):
        model.train()
        for x, y in generator.train_loader:
            optimizer.zero_grad()
            y_pred = model(x)

            loss_value = loss(y_pred, y)
            loss_value.backward()
            optimizer.step()

        score = evaluate(model, generator)

        for callback in callbacks:
            callback(epoch, score)
            if callback.should_stop():
                return model

    return model
