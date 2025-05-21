from typing import Any, Dict, KeysView, List, overload

import torch
from numpy.typing import NDArray


def _parse_value(value: Any) -> Any:
    return value


class KeyEnableData(object):
    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = _parse_value(value)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.__dict__, key)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__

    @overload
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    @overload
    def __getitem__(self, index: int | slice) -> Any: ...

    def keys(self) -> List[KeysView]:
        return self.__dict__.keys()

    def to_dict(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.__dict__.get(key, default)

    def pop(self, key: str, default: Any | None = None) -> Any:
        return self.__dict__.pop(key, default)

    def set_array(
        self, key: str, value: NDArray | torch.Tensor, index: int | slice | None = None
    ) -> None:
        if index is None:
            self.__dict__[key] = value
        else:
            self.__dict__[key][index] = value
