from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ObservationType = np.ndarray | torch.Tensor
ActionType = np.ndarray | torch.Tensor
RewardType = np.ndarray | torch.Tensor
DoneType = np.ndarray | torch.Tensor
ValueType = np.ndarray | torch.Tensor


ObservationNDArray = np.ndarray
ActionNDArray = np.ndarray
RewardNDArray = np.ndarray
DoneNDArray = np.ndarray
ValueNDArray = np.ndarray

ObservationTensor = torch.Tensor
ActionTensor = torch.Tensor
RewardTensor = torch.Tensor
DoneTensor = torch.Tensor
ValueTensor = torch.Tensor


LossType = torch.Tensor
ModelType = nn.Module
OptimizerType = optim.Optimizer
TensorType = torch.Tensor
DeviceType = torch.device | str