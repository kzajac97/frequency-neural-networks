from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple, Union

import numpy as np


class AbstractPredictiveSequenceGenerator(ABC):
    """
    Class implements abstract method for Predictive Sequence Generator

    Definition of `Predictive Modelling` in dynamical system
    identification can be found in https://arxiv.org/abs/1902.00683
    """

    def __init__(
        self,
        input_window_size: int,
        output_window_size: int,
        shift: Optional[int] = None,
        test_size: Union[int, float] = 0.5,
        use_overlap_in_test: bool = False,
    ):
        """
        :param input_window_size: length of input time series slice in samples
        :param output_window_size: length of output time series slice in samples
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlapping samples are allowed in test sequences
        """
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self._shift = shift
        self.test_size = test_size
        self.use_overlap_in_test = use_overlap_in_test

    @property
    @abstractmethod
    def data(self) -> Sequence:
        """Abstract property holding time series"""
        ...

    @property
    @abstractmethod
    def feature_columns(self) -> Tuple[int, ...]:
        """Abstract property holding slice of columns used as features"""
        ...

    @property
    @abstractmethod
    def target_columns(self) -> Tuple[int, ...]:
        """Abstract property holding slice of columns used as targets"""
        ...

    @property
    def first_test_index(self) -> int:
        """Returns first index of test data set using ratio or absolute sample number"""
        if self.test_size <= 1:
            return int(len(self.data) * (1 - self.test_size))

        return int(len(self.data) - self.test_size)

    def _get_shift(self, test: bool) -> int:
        """Gets shift value tu use"""
        if not self._shift:
            return self.input_window_size + self.output_window_size  # no overlap

        if not test:
            return self._shift  # train

        if self.use_overlap_in_test:
            return self._shift  # test with overlap

        return self.input_window_size + self.output_window_size  # test with no overlap

    def split(self, values: np.array, test: bool = False) -> Tuple[np.array, np.array]:
        """Splits time series data into input output pairs"""
        features, targets = [], []
        shift = self._get_shift(test)

        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)

        for index in list(range(len(values)))[:: shift]:
            if index + self.input_window_size + self.output_window_size >= len(values):
                break  # break when window size longer than input
            features.append(values[index : index + self.input_window_size, self.feature_columns])

            targets.append(
                values[
                    index + self.input_window_size : index + self.input_window_size + self.output_window_size,
                    self.target_columns,
                ]
            )

        return np.asarray(features), np.asarray(targets)


class AbstractSimulationSequenceGenerator(ABC):
    """
    Class implements abstract method for Simulation Sequence Generator

    Definition of `Simulation Modelling` in dynamical system
    identification can be found in https://arxiv.org/abs/1902.00683
    """

    def __init__(
        self,
        window_size: int,
        shift: Optional[int] = None,
        test_size: Union[int, float] = 0.5,
        use_overlap_in_test: bool = False,
        mask_test_outputs: bool = False,
        n_unmasked_test_samples: int = 0
    ):
        """
        :param window_size: length of input time series slice in samples
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlapping samples are allowed in test sequences
        :param mask_test_outputs: if True sequences are masked during model test
        :param n_unmasked_test_samples: number of test samples to compute metrics for
        """
        self.window_size = window_size
        self._shift = shift
        self.test_size = test_size
        self.use_overlap_in_test = use_overlap_in_test
        self.mask_test_outputs = mask_test_outputs
        self.n_unmasked_test_samples = n_unmasked_test_samples

    @property
    @abstractmethod
    def data(self) -> Sequence:
        """Abstract property holding time series"""
        ...

    @property
    @abstractmethod
    def feature_columns(self) -> Tuple[int, ...]:
        """Abstract property holding slice of columns used as features"""
        ...

    @property
    @abstractmethod
    def target_columns(self) -> Tuple[int, ...]:
        """Abstract property holding slice of columns used as targets"""
        ...

    @property
    def first_test_index(self) -> int:
        """Returns first index of test data set using ratio or absolute sample number"""
        if self.test_size <= 1:
            return int(len(self.data) * (1 - self.test_size))

        return int(len(self.data) - self.test_size)

    def _get_shift(self, test: bool) -> int:
        """Gets shift value tu use"""
        if not self._shift:
            return self.window_size  # no overlap

        if not test:
            return self._shift  # train

        if self.use_overlap_in_test:
            return self._shift  # test with overlap

        return self.window_size  # test with no overlap

    def split(self, values: np.array, test: bool = False) -> Tuple[np.array, np.array]:
        """Splits time series data into input output pairs"""
        features, targets = [], []
        shift = self._get_shift(test)

        if values.ndim == 1:
            values = np.expand_dims(values, axis=1)

        for index in list(range(len(values)))[:: shift]:
            if index + self.window_size > len(values):
                break  # break when window size longer than input
            features.append(values[index: index + self.window_size, self.feature_columns])

            if test and self.mask_test_outputs:
                # mask target samples leaving only `n_unmasked_test_samples`
                n_masked_samples = self.window_size - self.n_unmasked_test_samples
                targets.append(values[index + n_masked_samples: index + self.window_size, self.target_columns])
            else:
                targets.append(values[index: index + self.window_size, self.target_columns])

        return np.asarray(features), np.asarray(targets)
