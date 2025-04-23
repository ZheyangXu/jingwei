import gymnasium as gym

from jingwei.data.batch import Batch
from jingwei.data.buffer.base import Buffer


class MockBuffer(Buffer):
    def __init__(self, buffer_size: int, observation_space, action_space) -> None:
        super().__init__(
            buffer_size=buffer_size, observation_space=observation_space, action_space=action_space
        )

    def sample(self, batch_size: int) -> Batch:
        raise NotImplementedError("Sample method is not implemented for MockBuffer.")

    def _get_batch(self, batch_indexies: list[int]) -> Batch:
        raise NotImplementedError("Get batch method is not implemented for MockBuffer.")

    def reset(self) -> int:
        return 0


def test_buffer() -> None:
    env = gym.make("CartPole-v1")
    buffer = MockBuffer(
        buffer_size=100, observation_space=env.observation_space, action_space=env.action_space
    )
    assert buffer.size == 0
    assert buffer.capacity() == 100
    assert buffer.observation_shape == (4,)
    assert buffer.action_shape == (2,)
    assert buffer.device == "cpu"
    assert buffer.num_envs == 1
    assert buffer.pos == 0
    assert not buffer.full
