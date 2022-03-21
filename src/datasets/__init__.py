from functools import cached_property
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from .mixins import ChunkMixin, TorchDataLoaderMixin
from .numpy import NumpySimulationSequenceGenerator, NumpyPredictiveSequenceGenerator
from .pandas import PandasSimulationSequenceGenerator, PandasPredictiveSequenceGenerator


class TorchPredictiveSequenceLoader(NumpyPredictiveSequenceGenerator, TorchDataLoaderMixin):
    __doc__ = "\n".join([NumpyPredictiveSequenceGenerator.__doc__, TorchDataLoaderMixin.__doc__])  # inherit docstrings

    def __init__(
        self,
        states: np.array,
        input_window_size: int,
        output_window_size: int,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """
        :param states: array containing time series of system states
        :param input_window_size: size of input window to the model
        :param output_window_size: size of output window to the model
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        :param device: torch device to use
        """
        super().__init__(
            states,
            input_window_size,
            output_window_size,
            test_size,
            shift,
            use_overlap_in_test,
        )

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


class TorchPredictiveSequenceFrameLoader(PandasPredictiveSequenceGenerator, TorchDataLoaderMixin):
    __doc__ = "\n".join([PandasPredictiveSequenceGenerator.__doc__, TorchDataLoaderMixin.__doc__])  # inherit docstrings

    def __init__(
        self,
        data: pd.DataFrame,
        input_window_size: int,
        output_window_size: int,
        feature_column_names: Optional[Tuple[str]] = None,
        target_column_names: Optional[Tuple[str]] = None,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """
        :param data: data frame containing time series of system states
        :param input_window_size: size of input window to the model
        :param output_window_size: size of output window to the model
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        :param device: torch device to use
        """
        super().__init__(
            data,
            input_window_size,
            output_window_size,
            feature_column_names,
            target_column_names,
            test_size,
            shift,
            use_overlap_in_test,
        )

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


class TorchSimulationSequenceLoader(NumpySimulationSequenceGenerator, TorchDataLoaderMixin):
    __doc__ = "\n".join([NumpySimulationSequenceGenerator.__doc__, TorchDataLoaderMixin.__doc__])  # inherit docstrings

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
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
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
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        :param device: torch device to use
        """
        super().__init__(
            inputs,
            outputs,
            window_size,
            test_size,
            shift,
            use_overlap_in_test,
            mask_test_outputs,
            n_unmasked_test_samples,
        )

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


class TorchSimulationSequenceFrameLoader(PandasSimulationSequenceGenerator, TorchDataLoaderMixin):
    __doc__ = "\n".join([PandasSimulationSequenceGenerator.__doc__, TorchDataLoaderMixin.__doc__])  # inherit docstrings

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        feature_column_names: Optional[Tuple[str]] = None,
        target_column_names: Optional[Tuple[str]] = None,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False,
        mask_test_outputs: bool = False,
        n_unmasked_test_samples: int = 0,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """
        :param data: DataFrame with named columns containing inputs and outputs
        :param window_size: window size in samples
        :param feature_column_names: names of columns to use as features
        :param target_column_names: names of columns to use as targets
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        :param mask_test_outputs: if True sequences are masked during model test
        :param n_unmasked_test_samples: number of test samples to compute metrics for
        :param batch_size: batch size to use in samples
        :param dtype: torch numeric data type
        :param device: torch device to use
        """
        super().__init__(
            data,
            window_size,
            feature_column_names,
            target_column_names,
            test_size,
            shift,
            use_overlap_in_test,
            mask_test_outputs,
            n_unmasked_test_samples,
        )

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


class NumpyChunkPredictiveSequenceGenerator(NumpyPredictiveSequenceGenerator, ChunkMixin):
    __doc__ = "\n".join([NumpyPredictiveSequenceGenerator.__doc__, ChunkMixin.__doc__])  # inherit docstrings

    def __init__(
        self,
        states: np.array,
        input_window_size: int,
        output_window_size: int,
        test_size: Union[int, float] = 0.5,
        shift: Optional[int] = None,
        use_overlap_in_test: bool = False,
        use_test_split: bool = True,
    ):
        """
        :param states: array containing time series of system states
        :param input_window_size: size of input window to the model
        :param output_window_size: size of output window to the model
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        :param use_test_split: if True use testing setting in split
        """
        super().__init__(
            states,
            input_window_size,
            output_window_size,
            test_size,
            shift,
            use_overlap_in_test,
        )

        self.use_test_split = use_test_split

    @cached_property
    def train(self):
        raise NotImplementedError("Cannot use train property in chunk mode!")

    @cached_property
    def test(self):
        raise NotImplementedError("Cannot use test property in chunk mode!")


class NumpyChunkSimulationSequenceGenerator(NumpySimulationSequenceGenerator, ChunkMixin):
    __doc__ = "\n".join([NumpySimulationSequenceGenerator.__doc__, ChunkMixin.__doc__])  # inherit docstrings

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
        use_test_split: bool = True,
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
        :param use_test_split: if True use testing setting in split
        """
        super().__init__(
            inputs,
            outputs,
            window_size,
            test_size,
            shift,
            use_overlap_in_test,
            mask_test_outputs,
            n_unmasked_test_samples,
        )

        self.use_test_split = use_test_split

    @cached_property
    def train(self):
        raise NotImplementedError("Cannot use train property in chunk mode!")

    @cached_property
    def test(self):
        raise NotImplementedError("Cannot use test property in chunk mode!")


__all__ = [
    NumpyPredictiveSequenceGenerator,
    PandasPredictiveSequenceGenerator,
    TorchPredictiveSequenceLoader,
    TorchPredictiveSequenceFrameLoader,
    NumpySimulationSequenceGenerator,
    PandasSimulationSequenceGenerator,
    TorchSimulationSequenceLoader,
    TorchSimulationSequenceFrameLoader,
    NumpyChunkPredictiveSequenceGenerator,
    NumpyChunkSimulationSequenceGenerator,
]
