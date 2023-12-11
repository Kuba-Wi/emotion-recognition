from dataclasses import dataclass
from collections.abc import Callable
from net import BaseNet


@dataclass(kw_only=True)
class Config:
    """Class for storing runner config."""

    net: BaseNet
    net_path: str = f"data/{net.__str__()}"
    batch_size: int = 4
