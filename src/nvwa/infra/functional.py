import gymnasium as gym
import numpy as np


def get_observation_shape(
    observation_space: gym.spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    if isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        # Number of discrete features
        return (len(observation_space.nvec),)
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Dict):
        return {key: get_observation_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dimension(action_space: gym.spaces.Space) -> int:
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return len(action_space.nvec)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return int(action_space.n)
    elif isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
