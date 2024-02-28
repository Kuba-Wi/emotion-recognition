from dataclasses import dataclass
from collections.abc import Callable
from net import BaseNet
from typing import Type, Any, List, Optional, Callable


@dataclass(kw_only=True)
class Config:
    classes: List
    net_model: BaseNet
    net_dir: str
    batch_size: int
    num_epochs: int
    optimizer: Any
    criterion: Any
    transform: Any
    custom_criterion_call: Any = None

