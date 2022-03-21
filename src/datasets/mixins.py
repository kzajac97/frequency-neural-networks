from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class TorchDataLoaderMixin(ABC):
    """Mixin adding torch Data Loaders to SequenceGenerators"""
    def __init__(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        """
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        :param device: torch device to run training on
        """
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

    @cached_property
    @abstractmethod
    def train(self) -> Sequence:
        """Must be inherited from SequenceGenerator"""
        ...

    @cached_property
    @abstractmethod
    def test(self) -> Sequence:
        """Must be inherited from SequenceGenerator"""
        ...

    def _prepare_dataset(self, features: np.array, targets: np.array) -> TensorDataset:
        """Creates tensor dataset with consistent data types"""
        features = torch.from_numpy(features).to(self.dtype).to(self.device)
        targets = torch.from_numpy(targets).to(self.dtype).to(self.device)

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


class ChunkMixin(ABC):
    """Mixing adding functionality to split time series dataset using any ratio"""
    def __init__(self, use_test_split: bool = True):
        """
        :param use_test_split: if True use testing setting in split
        """
        self.use_test_split = use_test_split

    @property
    @abstractmethod
    def data(self) -> Sequence:
        """Must be inherited from SequenceGenerator"""
        ...

    def split(self, values: np.array, test: bool = False) -> Tuple[np.array, ...]:
        """Must be inherited from SequenceGenerator"""
        ...

    def _chunk(self, indices: np.array) -> List:
        """Chunks data using given splitting indices"""
        start = 0
        slices = []

        for end in indices:
            slices.append(self.split(self.data[start:end], test=self.use_test_split))
            start = end

        return slices

    def chunk_by_ratios(self, ratios: Tuple[float, ...]) -> List:
        """Chunk data by ratios relative to its length"""
        if sum(ratios) != 1:
            raise ValueError(f"Ratios must add up to 1! {sum(ratios)} != 1")

        return self._chunk((np.asarray(ratios) * len(self.data)).astype(int))

    def chunk_by_samples(self, samples: Tuple[int, ...]) -> List:
        """Chunk data by sample indices to its length"""
        if sum(samples) != len(self.data):
            raise ValueError(f"Samples must divide Sequence! {sum(samples)} != {len(self.data)}")

        return self._chunk(np.asarray(samples).astype(int))
