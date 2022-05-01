import torch
import torch.nn as nn


class SpectralTimeSeriesConv(nn.Module):
    """
    Torch module performing spectral convolution using complex set of learnable parameters

    It works in following steps:
        * Apply FFT to input time series signal
        * Multiply by complex parameter matrix
        * Apply inverse FFT to produced outputs
    """

    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
    ):
        """
        :param n_input_time_steps: number of time steps in input
        :param n_output_time_steps: number of time steps to produce
        :param n_input_state_variables: system input dimensionality
        :param n_output_state_variables: system output dimensionality
        """
        super(SpectralTimeSeriesConv, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.input_fourier_modes = n_input_time_steps // 2 + 1  # number of frequency modes resulting from RFFT
        self.output_fourier_modes = n_output_time_steps // 2 + 1
        # random matrix of complex weights
        # TODO: Initializers as a parameter
        weights = torch.rand(
            self.input_fourier_modes, self.output_fourier_modes, self.n_output_state_variables, dtype=torch.cfloat
        )
        self.weights = nn.Parameter(weights)

    def complex_batch_multiply(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Variables in `torch.einsum` operation:
            :var b: batch
            :var i: input time steps
            :var o: output time steps
            :var x: system dimensions
        """
        return torch.einsum("bix,iox->box", inputs, self.weights)

    def _forward_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """RFFT transform of the input signal"""
        outputs = torch.fft.rfft(inputs, n=self.n_input_time_steps, dim=1)
        return outputs.to(torch.cfloat)  # keep consistent complex type

    def _backward_transform(self, inputs: torch.Tensor) -> torch.Tensor:
        """IRFFT transform of the model output"""
        outputs = torch.fft.irfft(inputs, n=self.n_output_time_steps, dim=1)
        return outputs.to(torch.float32)  # cast to keep consistency between float and double

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variables = self._forward_transform(inputs)
        variables = self.complex_batch_multiply(inputs=variables)
        outputs = torch.fft.irfft(variables, n=self.n_output_time_steps, dim=1)
        # TODO: Activation can be applied here

        return outputs.to(torch.float32)
