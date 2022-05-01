from typing import Any, Dict

import torch
from torch import nn

from .accessors import ACTIVATION_MAPPING


class LinearModel(nn.Module):
    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        n_hidden_units: int = 100,
        n_hidden_layers: int = 1,
        hidden_activation: str = "relu",
    ):
        """
        :param n_input_time_steps: Number of input time steps.
        :param n_output_time_steps: Number of output time steps.
        :param n_input_state_variables:  Number of input state variables.
        :param n_output_state_variables: Number of output state variables.
        :param n_hidden_units: Number of hidden units.
        :param n_hidden_layers: Number of hidden layers.
        :param hidden_activation: Activation function given as str.
        """
        super().__init__()

        self.flatten_layer = nn.Flatten()

        self.input_layer = nn.Sequential(*[
            nn.Linear(n_input_time_steps * n_input_state_variables, n_hidden_units),
            ACTIVATION_MAPPING[hidden_activation](),
        ])

        self.hidden_linear_stack = nn.Sequential(
            *[
                nn.Linear(n_hidden_units, n_hidden_units),
                ACTIVATION_MAPPING[hidden_activation](),
            ]
            * n_hidden_layers
        )

        self.output_layer = nn.Linear(n_hidden_units, n_output_time_steps * n_output_state_variables)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create a model from sweep configuration."""
        return cls(
            n_input_time_steps=config["n_input_time_steps"],
            n_output_time_steps=config["n_output_time_steps"],
            n_input_state_variables=config["n_input_state_variables"],
            n_output_state_variables=config["n_output_state_variables"],
            n_hidden_units=config["n_hidden_units"],
            n_hidden_layers=config["n_hidden_layers"],
            hidden_activation=config["hidden_activation"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.flatten_layer(x)
        x = self.input_layer(x)
        x = self.hidden_linear_stack(x)
        x = self.output_layer(x)
        return x.unsqueeze(dim=-1)
