from abc import ABC, abstractmethod

import gymnasium as gym
import torch

from nvwa.algorithm.base import Algorithm
from nvwa.infra.wrapper import DataWrapper


class BaseTrainer(ABC):
    def __init__(
        self,
        algo: Algorithm,
        env: gym.Env,
        buffer_size: int = 10000,
        max_epochs: int = 200,
        batch_size: int = 32,
        device: torch.device | str = torch.get_default_device(),
        dtype: torch.dtype = torch.float32,
        gradient_step: int = 1,
        eval_episode_count: int = 10,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.env = env
        self.buffer_size = buffer_size
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.wrapper = DataWrapper(self.env.observation_space, self.env.action_space, dtype, device)
        self.gradient_step = gradient_step
        self.eval_episode_count = eval_episode_count
        self._init_buffer()

    @abstractmethod
    def _init_buffer(self) -> None: ...

    @abstractmethod
    def rollout(self) -> int: ...

    def evaluate(self) -> float:
        total_reward = 0.0
        for _ in range(self.eval_episode_count):
            observation, _ = self.env.reset()
            done = False
            while not done:
                action = self.algo.get_action(
                    self.wrapper.wrap_observation(observation), deterministic=True
                )
                action = self.wrapper.unwrap_action(action)
                observation_next, reward, terminated, truncated, info = self.env.step(action)
                observation = observation_next
                total_reward += reward
                done = terminated or truncated

        return total_reward / self.eval_episode_count

    @abstractmethod
    def train(self) -> None: ...
