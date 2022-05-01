from typing import Optional

import torch
import torch.nn as nn


class NeuralFilter(nn.Module):
    """
    Operator learning a mapping from frequency space to output time series.

    It works in following steps:
        * Apply FFT to input time series signal
        * Truncate frequency modes
        * Apply transformation to produce system output
    """

    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        fourier_modes: Optional[int] = None,
    ):
        """
        :param n_input_time_steps: number of time steps in input
        :param n_output_time_steps: number of time steps to produce
        :param n_input_state_variables: system input dimensionality
        :param n_output_state_variables: system output dimensionality
        :param fourier_modes: number of relevant Fourier modes used to produce output
        """
        super(NeuralFilter, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables
        self.fourier_modes = fourier_modes if fourier_modes else n_input_time_steps // 2 + 1  # max modes if None given

        weights = torch.rand(self.n_output_time_steps, self.fourier_modes, dtype=torch.cfloat)
        self.weights = nn.Parameter(weights)

    def complex_multiply_to_output(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Variables in `torch.einsum` operation:
            :var b: batch
            :var f: frequencies
            :var t: time steps
            :var v: state variables
        """
        return torch.einsum("bfv,tf->btv", inputs, self.weights)

    def _forward_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """RFFT transform of the input signal"""
        outputs = torch.fft.rfft(inputs, n=self.n_input_time_steps, dim=1)
        return outputs.to(torch.cfloat)  # keep consistent complex type

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variables = self._forward_transform(inputs)
        variables = variables[:, : self.fourier_modes, :]  # truncate to relevant modes
        outputs = self.complex_multiply_to_output(variables)  # filter
        # TODO: Activation can be applied here

        return outputs.to(torch.float32)
