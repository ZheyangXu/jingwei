# -*- coding: UTF-8 -*-

from dataclasses import dataclass

from jingwei.infra.typing import *


@dataclass
class Transition(object):
    state: StateType
    action: ActionType
    reward: RewardType
    next_state: StateType
    done: bool
