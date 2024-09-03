import gymnasium as gym

from jingwei.domain.agent.base import BaseAgent
from jingwei.infra.buffer.base import BaseBuffer, TransitionBuffer
from jingwei.infra.data_wrapper import DataWrapper
from jingwei.infra.typing import *
from jingwei.transitions.base import Transition


class Rollout(object):
    def __init__(self, agent: BaseAgent, env: gym.Env, data_wrapper: DataWrapper) -> None:
        self.agent = agent
        self.env = env
        self.data_wrapper = data_wrapper

    def rollout(self, buffer: BaseBuffer, max_step: int = 100) -> int:
        observation, _ = self.env.reset()
        done = False
        num_data = 0

        for i in range(max_step):
            wrapped_observation = self.data_wrapper.observation_to_tensor(observation)
            action = self.agent.get_action(wrapped_observation)
            action = self.data_wrapper.unwrap_action(action)
            observation_next, reward, terminated, truncated, _ = self.env.step(action)
            buffer.push(
                Transition(observation, action, reward, observation_next, terminated, truncated)
            )
            done = terminated or truncated
            if done:
                break
            observation = observation_next
            num_data += 1
        return num_data
