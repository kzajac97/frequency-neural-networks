from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from .mixins import TorchDataLoaderMixin
from .numpy import NumpySimulationSequenceGenerator
from .pandas import PandasSimulationSequenceGenerator


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
        batch_size : int = 32,
        dtype: torch.dtype = torch.float32,
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
        """
        super().__init__(
            inputs,
            outputs,
            window_size,
            test_size,
            shift,
            use_overlap_in_test,
            mask_test_outputs,
            n_unmasked_test_samples
        )

        self.batch_size = batch_size
        self.dtype = dtype


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
        batch_size : int = 32,
        dtype: torch.dtype = torch.float32,
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
            n_unmasked_test_samples
        )

        self.batch_size = batch_size
        self.dtype = dtype


__all__ = [
    NumpySimulationSequenceGenerator,
    PandasSimulationSequenceGenerator,
    TorchSimulationSequenceLoader,
    TorchSimulationSequenceFrameLoader,
]
