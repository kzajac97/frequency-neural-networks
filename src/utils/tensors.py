from typing import List

import numpy as np
import torch


def detach_to_numpy(*tensors: torch.Tensor) -> List[np.array]:
    """Detaches the tensors from GPU and returns them as numpy arrays"""
    return [tensor.cpu().detach().numpy() for tensor in tensors]
