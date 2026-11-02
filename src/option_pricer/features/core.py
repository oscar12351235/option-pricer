"""Path features: path-dependent quantities written into the payoff state.

Each feature reads one simulated path (the spot at every grid date) and adds
named entries to the state dictionary that payoffs consume. ``provides`` lists
the keys a feature writes; the engine uses it to check that every key a payoff
requires is available, and evaluates only the features a payoff needs.

The engine has vectorised equivalents of the three built-in features for
whole path arrays; ``apply`` is the reference implementation, also used for
user-defined features.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np


class Feature(ABC):
    """Base class for path features."""

    @property
    @abstractmethod
    def provides(self) -> frozenset[str]:
        """State keys written by :meth:`apply`."""

    @abstractmethod
    def apply(self, path: Sequence[float] | np.ndarray, state: dict[str, Any]) -> None:
        """Compute the feature on ``path`` and store it in ``state``."""


def _time_grid(state: dict[str, Any], path: Sequence[float] | np.ndarray) -> Any:
    timegrid = state.get("t_grid")
    if not isinstance(timegrid, list | tuple | np.ndarray):
        raise ValueError("state['t_grid'] must be a list, tuple or ndarray of times")
    if len(timegrid) != len(path):
        raise ValueError("state['t_grid'] length must equal len(path)")
    return timegrid


class AsianFeature(Feature):
    """Arithmetic average over all grid dates, including ``t = 0`` (key ``"average"``)."""

    @property
    def provides(self) -> frozenset[str]:
        return frozenset({"average"})

    def apply(self, path: Sequence[float] | np.ndarray, state: dict[str, Any]) -> None:
        state["average"] = float(sum(path) / len(path))


class BarrierFeature(Feature):
    """Discretely monitored barrier.

    Writes ``f"{name}_flag"`` (barrier hit on some grid date) and, with
    ``track_time``, ``f"{name}_time"`` (first hitting time, ``inf`` if never
    hit). ``direction`` is ``"up"`` or ``"down"``; ``strict`` selects ``>``/``<``
    instead of ``>=``/``<=``.
    """

    def __init__(
        self, name: str, level: float, direction: str, strict: bool, track_time: bool
    ) -> None:
        if direction not in ("up", "down"):
            raise ValueError("Direction should be 'up' or 'down'")
        self.name = name
        self.level = float(level)
        self.direction = direction
        self.strict = bool(strict)
        self.track_time = bool(track_time)

    @property
    def provides(self) -> frozenset[str]:
        keys = {f"{self.name}_flag"}
        if self.track_time:
            keys.add(f"{self.name}_time")
        return frozenset(keys)

    def _crossed(self, s: float) -> bool:
        if self.direction == "up":
            return s > self.level if self.strict else s >= self.level
        return s < self.level if self.strict else s <= self.level

    def apply(self, path: Sequence[float] | np.ndarray, state: dict[str, Any]) -> None:
        hit_idx = next((i for i, s in enumerate(path) if self._crossed(float(s))), None)
        state[f"{self.name}_flag"] = hit_idx is not None

        if self.track_time:
            if hit_idx is None:
                state[f"{self.name}_time"] = math.inf
            else:
                state[f"{self.name}_time"] = float(_time_grid(state, path)[hit_idx])


class LookbackFeature(Feature):
    """Running maximum and/or minimum of the path (keys ``"max"``, ``"min"``).

    ``mode`` is ``"max"``, ``"min"`` or ``"both"``. With ``track_time`` the
    first time each extremum is reached is stored as ``"max_time"`` /
    ``"min_time"``.
    """

    def __init__(self, mode: str, track_time: bool) -> None:
        if mode not in ("max", "min", "both"):
            raise ValueError("mode should be 'max', 'min' or 'both'")
        self.mode = mode
        self.track_time = bool(track_time)

    @property
    def provides(self) -> frozenset[str]:
        keys = set()
        if self.mode in ("max", "both"):
            keys.add("max")
            if self.track_time:
                keys.add("max_time")
        if self.mode in ("min", "both"):
            keys.add("min")
            if self.track_time:
                keys.add("min_time")
        return frozenset(keys)

    def apply(self, path: Sequence[float] | np.ndarray, state: dict[str, Any]) -> None:
        values = np.asarray(path, dtype=float)
        timegrid = _time_grid(state, path) if self.track_time else None

        if self.mode in ("max", "both"):
            max_idx = int(np.argmax(values))
            state["max"] = float(values[max_idx])
            if timegrid is not None:
                state["max_time"] = float(timegrid[max_idx])

        if self.mode in ("min", "both"):
            min_idx = int(np.argmin(values))
            state["min"] = float(values[min_idx])
            if timegrid is not None:
                state["min_time"] = float(timegrid[min_idx])
