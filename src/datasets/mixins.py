from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TorchDataLoaderMixin(ABC):
    """Mixin adding torch Data Loaders to SequenceGenerators"""
    def __init__(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        """
        self.batch_size = batch_size
        self.dtype = dtype

    @cached_property
    @abstractmethod
    def train(self):
        """Must be inherited from SequenceGenerator"""
        ...

    @cached_property
    @abstractmethod
    def test(self):
        """Must be inherited from SequenceGenerator"""
        ...

    def _prepare_dataset(self, features: np.array, targets: np.array) -> TensorDataset:
        """Creates tensor dataset with consistent data types"""
        features = torch.from_numpy(features).to(self.dtype)
        targets = torch.from_numpy(targets).to(self.dtype)

        return TensorDataset(features, targets)

    @cached_property
    def train_loader(self):
        """Iterable DataLoader object containing training data"""
        features, targets = self.train
        dataset = self._prepare_dataset(features, targets)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    @cached_property
    def test_loader(self):
        """Iterable DataLoader object containing test data"""
        features, targets = self.test
        dataset = self._prepare_dataset(features, targets)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
