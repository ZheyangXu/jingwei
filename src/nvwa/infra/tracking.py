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

    def tracking(
        self,
        event: Dict[str, float],
        episode: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> None:
        """
        Track an event with optional episode number.

        Args:
            event (Dict[str, float]): The event data to track.
            episode (Optional[int]): The episode number, if applicable.
            global_step (Optional[int]): The global step number, if applicable.
        """
        pass

    def append(self, key: str, value: float) -> None:
        """
        Append a value to the tracking system.

        Args:
            key (str): The key for the value.
            value (float): The value to append.
        """
        pass

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
