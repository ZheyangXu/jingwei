import gymnasium as gym

from jingwei.domain.agent import BaseAgent
from jingwei.domain.buffer import BaseBuffer
from jingwei.infra.data_wrapper import DataWrapper
from jingwei.infra.mtype import MType
from jingwei.infra.typing import *
from jingwei.rollout.base import Rollout
from jingwei.transitions.base import *
from jingwei.transitions.base import TensorTransitionBatch, TransitionBatch


class Trainner(object):
    def __init__(
        self,
        agent: BaseAgent,
        env: gym.Env,
        rollout: Rollout,
        buffer: BaseBuffer,
        data_wrapper: DataWrapper,
        max_epoch: int = 100,
        n_rollout_steps: int = 100,
        n_gradient_steps: int = 1,  # for onpolicy it should be 1
        n_step_per_gradient_step: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.agent = agent
        self.env = env
        self.rollout = rollout
        self.buffer = buffer
        self.data_wrapper = data_wrapper
        self.max_epoch = max_epoch
        self.n_rollout_steps = n_rollout_steps
        self.n_gradient_steps = n_gradient_steps
        self.batch_size = batch_size

    def train(self) -> None:
        for epoch in range(self.max_epoch):
            # 1. rollout n transitions into buffer
            # for onpolicy it should be rollout_buffer
            # for offpolicy it should be replay_buffer
            self.rollout.rollout(self.buffer, self.n_rollout_steps)

            # 2. training step
            # for onpolicy it gradient step should be 1
            #     for batch in self.buffer():
            #         self.agent.update_step
            # for offpolicy it can be great than 1
            #     for step in gradient_step:
            #         batch = self.replay_buffer.sample()
            #         self.agent.update_step()
            if self.agent.mtype == MType.off_policy:
                n_gradient_steps = self.n_gradient_steps
            else:
                n_gradient_steps = int(self.buffer.len() / self.batch_size)

            for i in range(n_gradient_steps):
                batch = self.buffer.get(self.batch_size)
                wrapped_batch = self.data_wrapper.to_tensor_transition(batch)
                self.agent.update_step(wrapped_batch)

    def evaluation_step(self) -> None:
        pass
