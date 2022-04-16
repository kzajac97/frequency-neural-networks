import torch
import torch.nn as nn


class MaskWarmupWrapper(nn.Module):
    """
    Wrapper masking warmup sample during inference

    It can be used in simulation modelling and it allows model to use more data during training,
    but output only relevant part of predicted time series during inference
    """

    def __init__(self, model: nn.Module, n_unmasked_samples: int):
        """
        :param model: Torch model returning 3D tensors with shape (BATCH_SIZE, TIME_STEPS, STATE_VARIABLES)
        :param n_unmasked_samples: number of unmasked samples during inference
        """
        super(MaskWarmupWrapper, self).__init__()

        self.model = model
        self.n_unmasked_samples = n_unmasked_samples

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.model(inputs)

        return self.model(inputs)[:, -self.n_unmasked_samples:, :]
