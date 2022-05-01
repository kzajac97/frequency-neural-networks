from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


class AbstractCallback(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @abstractmethod
    def should_stop(self) -> bool:
        ...


class LoggerCallback(AbstractCallback):
    def __init__(
        self,
        log_frequency: int = 1,
        metrics: List[str] = None,
        precision: int = 4,
        width: int = 4,
        separator: str = " ",
    ):
        """
        Callback that logs the loss and accuracy of the model every `log_frequency` epochs.

        :param log_frequency: The frequency with which to log the loss and accuracy.
        :param metrics: The metrics to log.
        :param precision: The precision to use when logging the metrics.
        :param width: Spacing between logged metrics.
        :param separator: The separator to use between logged metrics.
        """
        super().__init__()

        self.log_frequency = log_frequency
        self.metrics = metrics
        self.precision = precision
        self.width = width
        self.separator = separator

    def _format_metrics(self, metrics: pd.Series) -> str:
        """Formats the metrics for logging."""
        spacing = self.separator * self.width
        return f"{spacing}".join(
            [f"{metric_name}: {metric:.{self.precision}f}" for metric_name, metric in metrics.to_dict().items()]
        )

    def __call__(self, epoch: int, regression_score: pd.Series) -> None:
        if epoch % self.log_frequency == 0:
            print(f"Epoch: {epoch}{self.separator * self.width}{self._format_metrics(regression_score[self.metrics])}")

    def should_stop(self) -> bool:
        """Always returns False."""
        return False


class EarlyStoppingCallback(AbstractCallback):
    def __init__(self, patience: int = 5, delta: float = 0.00, metric_name: str = "MSE"):
        """
        :param patience: The number of epochs to wait before stopping.
        :param delta: The minimum change in the metric to trigger an early stopping.
        :param metric_name: The metric to use for early stopping.
        """
        super().__init__()

        self.patience = patience
        self.delta = delta
        self.metric_name = metric_name

        self.scores = []

    def __call__(self, epoch: int, regression_score: pd.Series) -> None:
        self.scores.append(regression_score[self.metric_name])

    def should_stop(self) -> bool:
        """Returns True if the model should stop training."""
        if len(self.scores) < self.patience:
            return False

        return (np.diff(self.scores[-self.patience :]) >= self.delta).all()
