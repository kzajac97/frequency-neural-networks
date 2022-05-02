import numpy as np
import pandas as pd
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in neural model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def describe_parameter_values(model: nn.Module, precision: int = 4) -> None:
    """Print statistics of the values of all parameters in the neural model"""
    return (
        pd.Series(np.concatenate([param.detach().numpy().flatten() for param in model.parameters()]))
        .describe()
        .apply(round, args=(precision,))
    )
