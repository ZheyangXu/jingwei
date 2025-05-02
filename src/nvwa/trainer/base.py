from abc import ABC, abstractmethod

import gymnasium as gym
import torch
import numpy as np


from nvwa.data.batch import Batch
from nvwa.data.buffer import ReplayBuffer
from nvwa.data.transition import Transition
from nvwa.infra.wrapper import DataWrapper
from nvwa.algorithm.base import OffPolicyAlgorithm


class OffPolicyTrainer(object):
    def __init__(
        self,
        algo: OffPolicyAlgorithm,
        env: gym.Env,
        buffer_size: int = 10000,
        minimal_size: int = 320,
        max_epochs: int = 200,
        batch_size: int = 32,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.algo = algo
        self.env = env
        self.buffer_size = buffer_size
        self.buffer = self._init_buffer()
        self.max_epochs = max_epochs
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.wrapper = DataWrapper(self.env.observation_space, self.env.action_space, dtype, device)
        self.gradient_step = 1

    def _init_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.buffer_size, self.env.observation_space, self.env.action_space)

    def rollout(self) -> int:
        num_transitions = 0
        observation, _ = self.env.reset()
        observation = self.wrapper.wrapper_observation(observation)
        while num_transitions < self.minimal_size:
            action = self.algo.get_action_from_behavior(observation)
            observation_next, reward, terminated, truncated, info = self.env.step(
                self.wrapper.unwrap_action(action)
            )
            observation_next = self.wrapper.wrap_observation(observation_next)
            transition = Transition(
                observation_tensor, action, reward, observation_next, terminated
            )
            self.buffer.put(transition)
            observation = observation_next
            num_transitions += 1
            if terminated or truncated:
                observation, info = self.env.reset()
                observation_tensor = self.wrapper.wrap_observation(observation)

        return self.buffer.size()

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.rollout()
            for i in range(self.gradient_step):
                if self.buffer.size() < self.minimal_size:
                    break
                batch = self.buffer.sample(self.batch_size)
                self.algo.update(batch)
