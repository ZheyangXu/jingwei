import numpy as np
import torch

from jingwei.infra.functional.torch import to_numpy, to_tensor


def test_to_tensor() -> None:
    ndarr = np.array([1, 2, 3])
    tensor = to_tensor(ndarr)
    cpu_float16_tensor = to_tensor(ndarr, dtype=torch.float16, device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert torch.allclose(tensor, torch.tensor([1, 2, 3], dtype=torch.float32))
    assert torch.allclose(
        cpu_float16_tensor, torch.tensor([1, 2, 3], dtype=torch.float16, device="cpu")
    )


def test_to_numpy() -> None:
    tensor = torch.tensor([1, 2, 3])
    ndarr = to_numpy(tensor)
    int_ndarr = to_numpy(tensor, dtype=np.int32)
    assert isinstance(ndarr, np.ndarray)
    np.testing.assert_allclose(ndarr, np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_allclose(int_ndarr, np.array([1, 2, 3], dtype=np.int32))
