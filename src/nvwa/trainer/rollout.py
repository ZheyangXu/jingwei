from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from nvwa.algorithm.base import Algorithm
from nvwa.data.batch import RolloutBatch
from nvwa.infra.functional import get_action_dimension, get_observation_shape
from nvwa.infra.wrapper import DataWrapper


class Rollout(object):
    def __init__(
        self,
        algo: Algorithm,
        env: gym.Env,
        max_size: int = 10000,
        n_rollout_step: Optional[int] = None,
        n_episodes: int = 1,
        dtype: Optional[np.dtype] = None,
        device: torch.device | str = torch.device("cpu"),
    ) -> None:
        self.algo = algo
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.observation_shape = get_observation_shape(env.observation_space)
        self.action_dimension = get_action_dimension(env.action_space)
        self.max_size = max_size
        self.n_rollout_step = n_rollout_step
        self.n_episodes = n_episodes
        self.dtype = dtype if dtype is not None else env.observation_space.dtype
        self.action_dtype: None | np.dtype[np.Any] | np.dtype[np.void] = env.action_space.dtype
        self.reset()
        self.device = device
        self.wrapper = DataWrapper(
            self.env.observation_space, self.env.action_space, torch.float32, self.device
        )
        print(
            f"Rollout initialized with max_size: {self.max_size}, n_rollout_step: {self.n_rollout_step}"
        )

    def data_keys(self) -> List[str]:
        return [
            "observation",
            "action",
            "reward",
            "observation_next",
            "terminated",
            "truncated",
            "episode_index",
        ]

    def get(self, key: str, index: Optional[slice | int] = None) -> np.ndarray:
        if key not in self.data_keys():
            raise KeyError(f"Invalid key: {key}. Available keys are: {self.data_keys()}")
        data = getattr(self, key)
        return data[index] if index is not None else data

    def reset(self) -> None:
        self.observation = np.zeros((self.max_size, *self.observation_shape), dtype=self.dtype)
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action = np.zeros((self.max_size, 1), dtype=self.action_dtype)
        else:
            self.action = np.zeros((self.max_size, self.action_dimension), dtype=self.action_dtype)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.observation_next = np.zeros_like(self.observation)
        self.terminated = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.truncated = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.episode_index = np.zeros((self.max_size,), dtype=np.int64)
        self.episode_end_positions = []
        self.pos = 0

    def rollout(self) -> RolloutBatch:
        self.reset()
        total_reward = 0.0
        for episode in range(self.n_episodes):
            observation, _ = self.env.reset()
            terminated = False
            truncated = False
            while True:
                if self._should_stop_rollout(terminated, truncated, self.pos):
                    self.episode_end_positions.append(self.pos)
                    break
                action = self.algo.get_action(self.wrapper.wrap_observation(observation), False)

                action = self.wrapper.unwrap_action(action)
                observation_next, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                self.observation[self.pos] = observation
                self.action[self.pos] = action
                self.reward[self.pos] = reward
                self.observation_next[self.pos] = observation_next
                self.terminated[self.pos] = terminated
                self.truncated[self.pos] = truncated
                self.episode_index[self.pos] = episode
                self.pos += 1
                observation = observation_next

        return self.to_batch()

    def to_batch(self) -> RolloutBatch:
        batch = RolloutBatch(
            observation=self.observation[: self.pos],
            action=self.action[: self.pos],
            reward=self.reward[: self.pos],
            observation_next=self.observation_next[: self.pos],
            terminated=self.terminated[: self.pos],
            truncated=self.truncated[: self.pos],
            episode_index=self.episode_index[: self.pos],
            _episode_end_position=self.episode_end_positions,
        )
        return batch

    def _should_stop_rollout(self, terminated: bool, truncated: bool, pos: int) -> bool:
        if terminated or truncated:
            return True
        if self.n_rollout_step is not None and pos >= self.n_rollout_step:
            return True
        if pos >= self.max_size:
            return True
        return False
