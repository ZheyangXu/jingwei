from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ObservationType = Type[np.ndarray | torch.Tensor]
ActionType = Type[np.ndarray | torch.Tensor]
RewardType = Type[np.ndarray | torch.Tensor]
DoneType = Type[np.ndarray | torch.Tensor]
ValueType = Type[np.ndarray | torch.Tensor]


ObservationNDArray = Type[np.ndarray]
ActionNDArray = Type[np.ndarray]
RewardNDArray = Type[np.ndarray]
DoneNDArray = Type[np.ndarray]
ValueNDArray = Type[np.ndarray]

ObservationTensor = Type[torch.Tensor]
ActionTensor = Type[torch.Tensor]
RewardTensor = Type[torch.Tensor]
DoneTensor = Type[torch.Tensor]
ValueTensor = Type[torch.Tensor]


LossType = Type[torch.Tensor]
ModelType = Type[nn.Module]
OptimizerType = Type[optim.Optimizer]
TensorType = Type[torch.Tensor]
DeviceType = Type[torch.device | str]