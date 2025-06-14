from ast import List
from typing import Dict, Iterable, Optional


class Tracking(object):
    def __init__(self) -> None:
        self._episode = 0
        self._global_step = 0

    def __getattr__(self, key: str) -> List[float]:
        if key not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        return self.__dict__[key]

    def __setattr__(self, key: str, value: List[float]) -> None:
        self.__dict__[key] = value

    def tracking(self, event: Dict[str, float]) -> None:
        """
        Track an event with optional episode number.

        Args:
            event (Dict[str, float]): The event data to track.
            episode (Optional[int]): The episode number, if applicable.
            global_step (Optional[int]): The global step number, if applicable.
        """
        for key, value in event.items():
            self.append(key, value)

    def append(self, key: str, value: float) -> None:
        """
        Append a value to the tracking system.

        Args:
            key (str): The key for the value.
            value (float): The value to append.
        """
        if key not in self.__dict__:
            self.__dict__[key] = []
        self.__dict__[key].append(value)

    @property
    def episode(self) -> int:
        """
        Get the current episode number.

        Returns:
            int: The current episode number.
        """
        return self._episode

    @episode.setter
    def episode(self, value: int) -> None:
        """
        Set the current episode number.

        Args:
            value (int): The episode number to set.
        """
        self._episode = value

    @property
    def global_step(self) -> int:
        """
        Get the current global step number.

        Returns:
            int: The current global step number.
        """
        return self._global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        """
        Set the current global step number.

        Args:
            value (int): The global step number to set.
        """
        self._global_step = value

    def step(self) -> None:
        self._global_step += 1

    def episode_step(self) -> None:
        self._episode += 1

    def to_dict(self) -> Dict[str, float]:
        """
        Convert the tracking data to a dictionary.

        Returns:
            Dict[str, float]: The tracking data as a dictionary.
        """
        return self.__dict__

    def reset(self) -> None:
        """
        Reset the tracking data.
        """
        self._episode = 0
        self._global_step = 0
        self.__dict__.clear()
        self.__init__()

    def keys(self) -> Iterable[str]:
        """
        Get the keys of the tracking data.

        Returns:
            List[str]: The keys of the tracking data.
        """
        return self.__dict__.keys()
