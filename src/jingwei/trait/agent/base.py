from copy import deepcopy
from typing import Any, Callable

from jingwei.domain.critic import BaseAgent
from jingwei.domain.buffer import BaseBuffer
from jingwei.infra.typing import *
from jingwei.trait.actor.base import ActorTrait
from jingwei.trait.critic.base import CriticTrait
from jingwei.transitions.base import TransitionBatch


class AgentTrait(BaseAgent):
    def __init__(
        self, actor: ActorTrait, estimate_return_fn: Callable, actor_loss_fn: Callable
    ) -> None:
        super().__init__()
        self.actor = actor
        self.estimate_return_fn = estimate_return_fn
        self.actor_loss_fn = actor_loss_fn

    def get_action(self, observation: ObservationTensor) -> ActionTensor:
        return self.get_action(observation)

    def estimate_return(self, transitions: TransitionBatch) -> ValueTensor:
        return self.estimate_return_fn(transitions)

    def update_step(self, transitions: TransitionBatch) -> None:
        actor_loss = self.actor_loss_fn(transitions)
        self.actor.update_step(actor_loss)


class ActorCriticAgentTrait(AgentTrait):
    def __init__(
        self,
        actor: ActorTrait,
        critic: CriticTrait,
        estimate_return_fn: Callable[..., Any],
        actor_loss_fn: Callable[..., Any],
        critic_loss_fn: Callable[..., Any],
    ) -> None:
        super().__init__(actor, estimate_return_fn, actor_loss_fn)
        self.critic = critic
        self.critic_loss_fn = critic_loss_fn
