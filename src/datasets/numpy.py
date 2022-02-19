from functools import cached_property
from typing import Optional, Union, Tuple

import numpy as np

from .abstract import AbstractPredictiveSequenceGenerator, AbstractSimulationSequenceGenerator


class NumpyPredictiveSequenceGenerator(AbstractPredictiveSequenceGenerator):
    """Class holds implementation of Sequence generator for dynamical system simulation problems"""
    def __init__(
        self,
        states: np.array,
        input_window_size: int,
        output_window_size: int,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False
    ):
        """
        :param states: array containing time series of system states
        :param input_window_size: size of input window to the model
        :param output_window_size: size of output window to the model
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        """
        super().__init__(input_window_size, output_window_size, shift, test_size, use_overlap_in_test)

        self.states = states
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self._shift = shift
        self.use_overlap_in_test = use_overlap_in_test

    @property
    def data(self) -> np.array:
        """Array of system states"""
        if self.states.ndim == 1:
            return np.expand_dims(self.states, axis=-1)  # expand to at least 2D

        return self.states

    @property
    def feature_columns(self) -> slice:
        """Slice of array to use as features"""
        return slice(None, self.states.shape[-1])

    @property
    def target_columns(self) -> slice:
        """Slice of array to use as targets"""
        return slice(self.states.shape[-1], None)

    @cached_property
    def train(self) -> Tuple[np.array, np.array]:
        """Cached train data split into sequences"""
        return self.split(self.data[:self.first_test_index], test=False)

    @cached_property
    def test(self) -> Tuple[np.array, np.array]:
        """Cached test data split into sequences"""
        return self.split(self.data[self.first_test_index:], test=True)


class NumpySimulationSequenceGenerator(AbstractSimulationSequenceGenerator):
    """Class holds implementation of Sequence generator for dynamical system simulation problems"""
    def __init__(
        self,
        inputs: np.array,
        outputs: np.array,
        window_size: int,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False,
        mask_test_outputs: bool = False,
        n_unmasked_test_samples: int = 0,
    ):
        """
        :param inputs: array of model inputs
        :param outputs: array of model target outputs
        :param window_size: window size in samples
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        :param mask_test_outputs: if True sequences are masked during model test
        :param n_unmasked_test_samples: number of test samples to compute metrics for
        """
        super().__init__(window_size, shift, test_size, use_overlap_in_test, mask_test_outputs, n_unmasked_test_samples)

        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> np.array:
        if self._inputs.ndim == 1:
            return np.expand_dims(self._inputs, axis=-1)  # expand to at least 2D

        return self._inputs

    @property
    def outputs(self) -> np.array:
        if self._outputs.ndim == 1:
            return np.expand_dims(self._outputs, axis=-1)  # expand to at least 2D

        return self._inputs

    @property
    def data(self) -> np.array:
        """Stacked inputs and outputs, required to generate split data"""
        return np.column_stack([self.inputs, self.outputs])

    @property
    def feature_columns(self) -> slice:
        """Slice of array to use as features"""
        return slice(None, self.inputs.shape[-1])

    @property
    def target_columns(self) -> slice:
        """Slice of array to use as targets"""
        return slice(self.inputs.shape[-1], None)

    @cached_property
    def train(self) -> Tuple[np.array, np.array]:
        """Cached train data split into sequences"""
        return self.split(self.data[:self.first_test_index], test=False)

    @cached_property
    def test(self) -> Tuple[np.array, np.array]:
        """Cached test data split into sequences"""
        return self.split(self.data[self.first_test_index:], test=True)
