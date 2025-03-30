import gymnasium as gym


def get_space_shape(space: gym.spaces.Space) -> tuple[int, ...]:
    """Get the shape of the observation space.

    Args:
        space (gym.spaces.Space): The observation space.

    Returns:
        tuple[int, ...]: The shape of the observation space.
    """
    if isinstance(space, gym.spaces.Box):
        return space.shape
    elif isinstance(space, gym.spaces.Discrete):
        return (space.n,)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return (*space.nvec,)
    elif isinstance(space, gym.spaces.MultiBinary):
        return space.shape
    else:
        raise ValueError(f"Unsupported observation space type: {type(space)}")
