from enum import Enum


class MType(Enum):
    off_policy = "off_policy"
    on_policy = "on_policy"
    off_line = "off_line"
