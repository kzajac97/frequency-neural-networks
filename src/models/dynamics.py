import torch

from dynonet.functional import MimoLinearDynamicalOperatorFun
from utils.tensors import detach_to_numpy


def lowpass_filter_from_components(numerator, denominator, inputs):
    # TODO: Implement lowpass filter using torchaudio
    #       https://pytorch.org/audio/stable/functional.html?highlight=lfilt#torchaudio.functional.lfilter
    ...


class AutogradLowpassFilter(torch.autograd.Function):
    """
    Reimplementation of the autograd compatible lowpass filter from the paper:
    "dynoNet: A neural network architecture for learning dynamical systems"

    For details refer to the paper: https://arxiv.org/pdf/2006.02250.pdf

    Note: Some variables have different names to align to configuration used by other models
          Code style can be slightly changed to align to CI pipelines
    """
    @staticmethod
    def forward(context, numerator, denominator, inputs):
        # TODO: Implement the forward pass
        ...

    @staticmethod
    def backward():
        # TODO: Implement the backward pass
        ...


class LearnableDynamicalOperator(torch.nn.Module):
    """
    Reimplementation of the learnable dynamical operator from the paper:
    "dynoNet: A neural network architecture for learning dynamical systems"

    For details refer to the paper: https://arxiv.org/pdf/2006.02250.pdf

    Note: Some variables have different names to align to configuration used by other models
          Code style can be slightly changed to align to CI pipelines
    """
    def __init__(
        self,
        n_input_state_variables: int,  # for SISO just pass 1
        n_output_state_variables: int,
        n_numerator_parameters: int,
        n_denominator_parameter: int,
        init_range: float = 0.01,
    ):
        """
        :param n_input_state_variables: Number of input state variables
        :param n_output_state_variables: Number of output state variables
        :param n_numerator_parameters: Number of numerator parameters in filtering polynomial
        :param n_denominator_parameter: Number of denominator parameters in filtering polynomial
        :param init_range: initialization range, default value from paper
        """
        super().__init__()
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables
        self.n_numerator_parameters = n_numerator_parameters
        self.n_denominator_parameter = n_denominator_parameter

        self.numerator_parameters = torch.nn.Parameter(
            (torch.rand(self.n_input_state_variables, self.n_output_state_variables, self.n_numerator_parameters) - 0.5) * 2 * init_range
        )

        self.denominator_parameters = torch.nn.Parameter(
            (torch.rand(self.n_input_state_variables, self.n_output_state_variables, self.n_denominator_parameter) - 0.5) * 2 * init_range
        )

    def detach_parameters(self):
        return detach_to_numpy(self.numerator_parameters.data, self.denominator_parameters.data)

    def get_filter_data(self):
        """Returns the numerator and denominator coefficients of the filter"""
        return (
            torch.cat([torch.zeros(1), self.numerator_parameters.data.flatten()]),
            torch.cat([torch.ones(1), self.denominator_parameters.data.flatten()])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = AutogradLowpassFilter.apply(self.denominator_parameters, self.numerator_parameters, inputs)
        return outputs.to(inputs.device)


class StaticNonLinearity(torch.nn.Module):
    """
    Reimplementation of the static non-linearity from the paper:
    "dynoNet: A neural network architecture for learning dynamical systems"

    For details refer to the paper: https://arxiv.org/pdf/2006.02250.pdf

    Note: Some variables have different names to align to configuration used by other models
          Code style can be slightly changed to align to CI pipelines
    """

    def __init__(
        self,
        n_input_state_variables: int,
        n_output_state_variables: int,
        n_hidden_units: int,
        activation: torch.nn.Module,
    ):
        """
        :param n_input_state_variables: Number of input state variables
        :param n_output_state_variables: Number of output state variables
        :param n_hidden_units: Number of hidden units
        :param activation: Activation function
        """
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_state_variables, n_hidden_units),
            activation(),
            torch.nn.Linear(n_hidden_units, n_output_state_variables)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
