import torch

from nvwa.data.batch import Batch


dtype = torch.float32
device = torch.device("cpu")


def test_batch() -> None:
    observation = torch.randn(4, 32, 32, dtype=dtype, device=device)
    action = torch.randint(0, 2, (4,), dtype=dtype, device=device)
    reward = torch.randn(4, dtype=dtype, device=device)
    observation_next = torch.randn(4, 32, 32, dtype=dtype, device=device)
    terminated = torch.randint(0, 2, (4,), dtype=torch.int64, device=device)
    truncated = torch.randint(0, 2, (4,), dtype=torch.int64, device=device)
    batch = Batch(
        observation=observation,
        action=action,
        reward=reward,
        observation_next=observation_next,
        terminated=terminated,
        truncated=truncated,
    )

    assert isinstance(batch.observation, torch.Tensor)
    assert isinstance(batch.action, torch.Tensor)
    assert isinstance(batch.reward, torch.Tensor)
    assert isinstance(batch.observation_next, torch.Tensor)
    assert isinstance(batch.terminated, torch.Tensor)
    assert isinstance(batch.truncated, torch.Tensor)
    assert len(batch) == 4
    assert batch.observation_dtype() == torch.float32
    assert batch.action_dtype() == torch.float32
    assert batch.device() == device
