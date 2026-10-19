from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class PathSampler(Protocol):
    def __call__(
        self,
        n_paths: int,
        timegrid,
        randnumbergenerator,
        randnumber_key: str | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray: ...


class Model(ABC):
    @abstractmethod
    def s0(self) -> float: ...

    @abstractmethod
    def discount(self, t: float) -> float: ...

    @abstractmethod
    def path_sampler(self) -> PathSampler: ...
