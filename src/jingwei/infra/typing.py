# -*- coding: UTF-8 -*-

from typing import  Any, Dict, List, Tuple, Type, Union

import torch
import numpy as np


StateType = Type[np.ndarray]
ObservationType = Type[np.ndarray]
ObservationsType = Type[List[ObservationType]]
ValueType = Type[float]
AdvantageType = Type[float]
ActionType = Type[Union[int, float, np.ndarray]]
RewardType = Type[float]
LossType = Type[Union[float, torch.Tensor]]
