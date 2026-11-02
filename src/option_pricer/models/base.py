"""Interfaces shared by all models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class PathSampler(Protocol):
    """Simulates spot paths on a time grid.

    Returns an array of shape ``(n_paths, len(timegrid))`` whose first column
    is the initial spot. Samplers that set ``supports_crn = True`` accept a
    tensor of standard normals ``z`` of shape ``(n_steps, n_paths, n_factors)``
    supplied by the engine; otherwise ``z`` is ignored and all randomness comes
    from ``randnumbergenerator``.
    """

    def __call__(
        self,
        n_paths: int,
        timegrid: np.ndarray,
        randnumbergenerator: np.random.Generator,
        randnumber_key: str | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray: ...


class Model(ABC):
    """A pricing model: initial spot, discount curve and a path sampler."""

    @abstractmethod
    def s0(self) -> float:
        """Initial spot price."""

    @abstractmethod
    def discount(self, t: float) -> float:
        """Discount factor from time ``t`` to 0."""

    @abstractmethod
    def path_sampler(self) -> PathSampler:
        """Sampler used by the engine to simulate paths."""
