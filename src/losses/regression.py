import torch


class LogCoshLoss(torch.nn.Module):
    """LogCoshLoss regression loss function."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        """Computes the logarithm of the hyperbolic cosine of x."""
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self._log_cosh(y_pred - y_true)
