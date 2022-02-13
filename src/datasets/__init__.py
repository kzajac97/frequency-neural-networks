from typing import Optional, Union

import numpy as np
import torch

from .generators import SimulationSequenceGenerator
from .mixins import TorchDataLoaderMixin


class TorchSimulationSequenceGenerator(SimulationSequenceGenerator, TorchDataLoaderMixin):
    __doc__ = "\n".join([SimulationSequenceGenerator.__doc__, TorchDataLoaderMixin.__doc__])  # inherit docstrings

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


__all__ = [
    SimulationSequenceGenerator,
    TorchSimulationSequenceGenerator,
]
