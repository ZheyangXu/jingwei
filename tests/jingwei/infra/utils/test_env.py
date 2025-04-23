import gymnasium as gym

from jingwei.infra.utils.env import get_space_shape


def test_get_space_shape() -> None:
    # Test with Box space
    box_space = gym.spaces.Box(low=0, high=1, shape=(3, 4), dtype=float)
    assert get_space_shape(box_space) == (3, 4)

    # Test with Discrete space
    discrete_space = gym.spaces.Discrete(5)
    assert get_space_shape(discrete_space) == (5,)

    # Test with MultiDiscrete space
    multi_discrete_space = gym.spaces.MultiDiscrete([2, 3])
    assert get_space_shape(multi_discrete_space) == (2, 3)

    # Test with MultiBinary space
    multi_binary_space = gym.spaces.MultiBinary(4)
    assert get_space_shape(multi_binary_space) == (4,)


def test_get_space_shape_invalid() -> None:
    space = gym.spaces.Tuple(
        (gym.spaces.Discrete(2), gym.spaces.Box(low=0, high=1, shape=(3,), dtype=float))
    )
    try:
        get_space_shape(space)
    except ValueError as e:
        assert str(e) == f"Unsupported observation space type: {type(space)}"
    else:
        assert False, "Expected ValueError not raised"
