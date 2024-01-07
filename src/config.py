from dataclasses import dataclass
from collections.abc import Callable
from net import BaseNet
from typing import Type, Any, List


@dataclass(kw_only=True)
class Config:
    """Class for storing runner config."""

    classes: List
    net: Type[BaseNet]
    batch_size: int
    num_epochs: int
    optimizer: int
    criterion: Any
    net_path: str
