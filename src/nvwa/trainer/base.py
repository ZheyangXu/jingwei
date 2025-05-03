from abc import ABC, abstractmethod

import gymnasium as gym
import torch
import numpy as np


from nvwa.data.batch import Batch
from nvwa.data.buffer import ReplayBuffer, RolloutBuffer
from nvwa.data.transition import RolloutTransition, Transition
from nvwa.infra.wrapper import DataWrapper
from nvwa.algorithm.base import OffPolicyAlgorithm, OnPolicyAlgorithm
from torch import Tensor


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
        gradient_step: int = 1,
        num_episodes_per_evaluation: int = 2,
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
        self.gradient_step = gradient_step
        self.num_episodes_per_evaluation = num_episodes_per_evaluation

    def _init_buffer(self) -> ReplayBuffer:
        return ReplayBuffer(self.buffer_size, self.env.observation_space, self.env.action_space)

    def rollout(self) -> int:
        num_transitions = 0
        observation, _ = self.env.reset()
        observation = self.wrapper.wrap_observation(observation)
        while num_transitions < self.minimal_size:
            action = self.algo.get_behavior_action(observation)
            observation_next, reward, terminated, truncated, info = self.env.step(
                self.wrapper.unwrap_action(action)
            )
            observation_next = self.wrapper.wrap_observation(observation_next)
            transition = Transition(
                observation, action, reward, observation_next, terminated, truncated
            )
            self.buffer.put(transition)
            observation = observation_next
            num_transitions += 1
            if terminated or truncated:
                observation, info = self.env.reset()
                observation = self.wrapper.wrap_observation(observation)

        return self.buffer.size()

    def evaluate(self) -> float:
        total_reward = 0.0
        for _ in range(self.num_episodes_per_evaluation):
            observation, _ = self.env.reset()
            observation = self.wrapper.wrap_observation(observation)
            done = False
            while not done:
                action = self.algo.get_action(observation)
                observation_next, reward, terminated, truncated, info = self.env.step(
                    self.wrapper.unwrap_action(action)
                )
                observation_next = self.wrapper.wrap_observation(observation_next)
                observation = observation_next
                total_reward += reward
                done = terminated or truncated

        return total_reward / self.num_episodes_per_evaluation

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.rollout()
            epoch_loss = 0.0
            for i in range(self.gradient_step):
                if self.buffer.size() < self.minimal_size:
                    break
                batch = self.buffer.sample(self.batch_size)
                state = self.algo.update(batch)
                epoch_loss += state["loss"]
            print(f"epoch {epoch}, loss {epoch_loss:.4f}")

            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                print(f"epoch {epoch}, evaluation reward: {eval_reward:.4f}")


class OnPolicyTrainer(object):
    def __init__(
        self,
        algo: OnPolicyAlgorithm,
        env: gym.Env,
        buffer_size: int = 1000,
        max_epochs: int = 200,
        batch_size: int = 32,
        n_rollout_steps=10,
        device: torch.device | str = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        num_episodes_per_evaluation: int = 2,
    ) -> None:
        self.algo = algo
        self.env = env
        self.buffer_size = buffer_size
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_rollout_steps = n_rollout_steps
        self.device = device
        self.dtype = dtype
        self.buffer = self._init_buffer()
        self.wrapper = DataWrapper(self.env.observation_space, self.env.action_space)
        self.num_episodes_per_evaluation = num_episodes_per_evaluation

    def _init_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(self.buffer_size, self.env.observation_space, self.env.action_space)

    def evaluate(self) -> float:
        total_reward = 0.0
        for _ in range(self.num_episodes_per_evaluation):
            observation, _ = self.env.reset()
            observation = self.wrapper.wrap_observation(observation)
            done = False
            while not done:
                action = self.algo.get_action(observation)
                observation_next, reward, terminated, truncated, info = self.env.step(
                    self.wrapper.unwrap_action(action)
                )
                observation_next = self.wrapper.wrap_observation(observation_next)
                observation = observation_next
                total_reward += reward
                done = terminated or truncated

        return total_reward / self.num_episodes_per_evaluation

    def rollout(self) -> int:
        self.buffer.reset()
        for n_steps in range(self.n_rollout_steps):
            observation, _ = self.env.reset()
            observation = self.wrapper.wrap_observation(observation)
            done = False
            while not done:
                action, values, log_probs = self.algo.estimate_value(observation)
                observation_next, reward, terminated, truncated, info = self.env.step(
                    self.wrapper.unwrap_action(action)
                )
                observation_next = self.wrapper.wrap_observation(observation_next)
                transition = RolloutTransition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    observation_next=observation_next,
                    terminated=terminated,
                    truncated=truncated,
                    values=values,
                    log_prob=log_probs,
                    prob=log_probs,
                    advantages=values,
                )
                self.buffer.put(transition)
                observation = observation_next
                n_steps += 1
                if terminated or truncated:
                    break
        return self.buffer.size()

    def train(self) -> None:
        for epoch in range(self.max_epochs):
            self.rollout()
            for batch in self.buffer.get_batch(self.batch_size):
                self.algo.update(batch)

            if epoch % 10 == 0:
                eval_reward = self.evaluate()
                print(f"epoch {epoch}, evaluation reward: {eval_reward:.4f}")
