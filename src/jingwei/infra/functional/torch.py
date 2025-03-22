import numpy as np
import torch
from numpy.typing import NDArray


def to_tensor(
    ndarr: NDArray,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = torch.device("cpu"),
) -> torch.Tensor:
    return torch.tensor(ndarr, dtype=dtype, device=device)


def to_numpy(tensor: torch.Tensor, dtype: np.dtype = np.float32) -> NDArray:
    return tensor.cpu().numpy().astype(dtype)
