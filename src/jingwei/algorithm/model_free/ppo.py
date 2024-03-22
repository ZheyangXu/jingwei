# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F

from jingwei.actor import BaseActor
from jingwei.critic import BaseCritic
from jingwei.infra.typing import ActionType, LossType, RewardType, StateType, ValueType


class PPO(object):
    def __init__(
        self,
        actor: BaseActor,
        critic: BaseCritic,
        gamma: float = 0.9,
        eps: float = 0.02,
        epochs: int = 500,
        lmbda: float = 0.2,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

    def compute_advantages(
        self,
        states: StateType,
        rewards: RewardType,
        actions: ActionType,
        next_states: StateType,
        dones: bool,
    ) -> ValueType:
        td_target = rewards + self.gamma * self.critic.estimate_return(next_states) * (
            1 - dones
        )
        td_delta = td_target - self.critic.estimate_return(next_states)
        advantage_list = []
        advantage = 0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return advantage_list

    def get_actor_loss(
        self,
        old_log_probs: torch.Tensor,
        states: StateType,
        actions: ActionType,
        advantages: ValueType,
    ) -> LossType:
        log_probs = self.actor.get_log_probs(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        return torch.mean(-torch.min(surr1, surr2))

    def get_critic_loss(self, states: StateType, td_target: torch.Tensor) -> LossType:
        return torch.mean(F.mse_loss(self.critic.estimate_return(states), td_target))

    def update(self, trajectories: any) -> None:
        pass
