import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in neural model"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
