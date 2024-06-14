from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ObservationType = Type[np.ndarray | torch.Tensor]
ActionType = Type[np.ndarray | torch.Tensor]
RewardType = Type[np.ndarray | torch.Tensor]
DoneType = Type[np.ndarray[bool] | torch.Tensor]
ValueType = Type[np.ndarray | torch.Tensor]


ObservationNDArray = Type[np.ndarray[float | int]]
ActionNDArray = Type[np.ndarray[float | int]]
RewardNDArray = Type[np.ndarray[float]]
DoneNDArray = Type[np.ndarray[bool]]
ValueNDArray = Type[np.ndarray[float]]

ObservationTensor = Type[torch.Tensor[float]]
ActionTensor = Type[torch.Tensor[float]]
RewardTensor = Type[torch.Tensor[float]]
DoneTensor = Type[torch.Tensor[float]]
ValueTensor = Type[torch.Tensor[float]]


LossType = Type[torch.Tensor]
ModelType = Type[nn.Module]
OptimizerType = Type[optim.Optimizer]
TensorType = Type[torch.Tensor]
