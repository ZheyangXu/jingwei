import numpy as np
import torch


class DataWrapper(object):
    def __init__(self, dtype: torch.dtype, device: torch.device = "cpu") -> None:
        self.dtype = dtype
        self.device = device

    def to_numpy(self, data: torch.Tensor) -> np.ndarray:
        return data.detach().cpu().numpy()

    def to_tensor(self, data: np.ndarray, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        return torch.as_tensor(data, dtype=dtype, device=device)
