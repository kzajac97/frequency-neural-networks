from functools import cached_property
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from .abstract import AbstractPredictiveSequenceGenerator, AbstractSimulationSequenceGenerator


class PandasPredictiveSequenceGenerator(AbstractPredictiveSequenceGenerator):
    """Class holds implementation of Sequence generator for dynamical system simulation problems"""

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
    ):
        """
        :param data: data frame containing time series of system states
        :param input_window_size: size of input window to the model
        :param output_window_size: size of output window to the model
        :param test_size: ratio or number of samples to use for testing
        :param shift: spacing between starts of following slices, if None given no overlapping samples are generated
        :param use_overlap_in_test: if False no overlap in test is allowed
        """
        super().__init__(input_window_size, output_window_size, test_size, shift, use_overlap_in_test)

        self._data = data
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.feature_column_names = feature_column_names
        self.target_column_names = target_column_names
        self._shift = shift
        self.use_overlap_in_test = use_overlap_in_test

    @property
    def data(self) -> pd.DataFrame:
        """Required to implement abstract property"""
        return self._data

    @cached_property
    def column_names_to_slices(self) -> Dict[str, int]:
        """Mapping of column names to numpy array slices"""
        return {name: index for index, name in enumerate(self.data.columns)}

    @property
    def feature_columns(self) -> List[int]:
        """Returns indices of column used as features"""
        feature_columns_names = (
            self.feature_column_names if self.feature_column_names is not None else list(self.data.columns)
        )
        return [self.column_names_to_slices[name] for name in feature_columns_names]

    @property
    def target_columns(self) -> List[int]:
        """Returns indices of column used as targets"""
        target_columns_names = (
            self.target_column_names if self.target_column_names is not None else list(self.data.columns)
        )
        return [self.column_names_to_slices[name] for name in target_columns_names]

    @cached_property
    def train(self) -> Tuple[np.array, np.array]:
        """Cached train data split into sequences"""
        return self.split(self.data[: self.first_test_index], test=False)

    @cached_property
    def test(self) -> Tuple[np.array, np.array]:
        """Cached test data split into sequences"""
        return self.split(self.data[self.first_test_index :], test=True)


class PandasSimulationSequenceGenerator(AbstractSimulationSequenceGenerator):
    """Class holds implementation of Sequence generator for dynamical system simulation problems"""

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
        """
        super().__init__(window_size, shift, test_size, use_overlap_in_test, mask_test_outputs, n_unmasked_test_samples)

        self._data = data
        self.feature_column_names = feature_column_names
        self.target_column_names = target_column_names
        self.test_size = test_size

    @property
    def data(self) -> pd.DataFrame:
        """Required to implement abstract property"""
        return self._data

    @property
    def inputs(self) -> pd.DataFrame:
        """Rewrite inputs property from SequenceGenerator to return names columns from SequenceGenerator"""
        return self.data[self.feature_column_names]

    @property
    def outputs(self) -> pd.DataFrame:
        """Rewrite inputs property from SequenceGenerator to return names columns from SequenceGenerator"""
        return self.data[self.target_column_names]

    @cached_property
    def column_names_to_slices(self) -> Dict[str, int]:
        """Mapping of column names to numpy array slices"""
        return {name: index for index, name in enumerate(self.data.columns)}

    @property
    def feature_columns(self) -> List[int]:
        """Returns indices of column used as features"""
        feature_columns_names = (
            self.feature_column_names if self.feature_column_names is not None else list(self.data.columns)
        )
        return [self.column_names_to_slices[name] for name in feature_columns_names]

    @property
    def target_columns(self) -> List[int]:
        """Returns indices of column used as targets"""
        target_columns_names = (
            self.target_column_names if self.target_column_names is not None else list(self.data.columns)
        )
        return [self.column_names_to_slices[name] for name in target_columns_names]

    @cached_property
    def train(self) -> Tuple[np.array, np.array]:
        """Cached train data split into sequences"""
        return self.split(self.data.iloc[: self.first_test_index].values, test=False)

    @cached_property
    def test(self) -> Tuple[np.array, np.array]:
        """Cached test data split into sequences"""
        return self.split(self.data.iloc[self.first_test_index :].values, test=True)
