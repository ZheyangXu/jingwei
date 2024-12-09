import numpy as np
import torch

from jingwei.infra.typing import TensorLike


def test_torch_tensor_is_tensor_like():
    assert isinstance(torch.tensor([1, 2, 3]), TensorLike)


def test_numpy_array_is_tensor_like():
    assert isinstance(np.array([1, 2, 3]), TensorLike)


def test_list_is_not_tensor_like():
    assert not isinstance([1, 2, 3], TensorLike)
