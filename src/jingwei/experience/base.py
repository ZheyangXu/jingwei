# -*- coding: UTF-8 -*-

import dataclasses
from dataclasses import dataclass
from typing import Optional, TypeVar

from jingwei.infra.typing import *

T = TypeVar("T", bound="Transition")


@dataclass(frozen=False)
class Transition(object):
    state: StateType
    action: ActionType
    reward: RewardType
    next_state: StateType
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to(self, device: torch.device) -> T:
        for f in dataclasses.fields(self.__class__):
            if getattr(self, f.name) is not None:
                super().__setattr__(
                    f.name, torch.as_tensor(getattr(self, f.name)).to(device)
                )
        return self

    @property
    def device(self) -> torch.device:
        return self.state.device
