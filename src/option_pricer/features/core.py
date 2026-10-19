from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class Feature(ABC):
    @property
    @abstractmethod
    def provides(self): ...

    @abstractmethod
    def apply(self, path, state: dict) -> None: ...


class AsianFeature(Feature):
    @property
    def provides(self):
        return frozenset({"average"})

    def apply(self, path, state):
        state["average"] = float(sum(path) / len(path))


class BarrierFeature(Feature):
    def __init__(self, name: str, level: float, direction: str, strict: bool, track_time: bool):
        if direction not in ("up", "down"):
            raise ValueError("Direction should be 'up' or 'down'")
        self.name = name
        self.level = float(level)
        self.direction = direction
        self.strict = bool(strict)
        self.track_time = bool(track_time)

    @property
    def provides(self):
        keys = {f"{self.name}_flag"}
        if self.track_time:
            keys.add(f"{self.name}_time")
        return frozenset(keys)

    def apply(self, path, state):
        hit_time_id = None

        if self.direction == "up":
            for i, s in enumerate(path):
                if float(s) > self.level if self.strict else float(s) >= self.level:
                    hit_time_id = i
                    break
        else:  # down
            for i, s in enumerate(path):
                if float(s) < self.level if self.strict else float(s) <= self.level:
                    hit_time_id = i
                    break

        state[f"{self.name}_flag"] = hit_time_id is not None

        if self.track_time:
            timegrid = state.get("t_grid")
            if hit_time_id is None:
                state[f"{self.name}_time"] = math.inf
            else:
                if len(timegrid) != len(path):
                    raise ValueError("state['t_grid'] length must equal len(path)")
                if not isinstance(timegrid, list | tuple | np.ndarray):
                    raise ValueError("state['t_grid'] must be list/tuple/ndarray")
                state[f"{self.name}_time"] = float(timegrid[hit_time_id])


class LookbackFeature(Feature):
    def __init__(self, mode: str, track_time: bool):
        if mode not in ("max", "min", "both"):
            raise ValueError("mode should be 'max', 'min' or 'both'")
        self.mode = mode
        self.track_time = bool(track_time)

    @property
    def provides(self):
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

    def apply(self, path, state):
        timegrid = state.get("t_grid")

        if self.mode in ("max", "both"):
            max_val = path[0]
            max_idx = 0
            for i, s in enumerate(path):
                if s > max_val:
                    max_val = s
                    max_idx = i
            state["max"] = float(max_val)
            if self.track_time:
                state["max_time"] = float(timegrid[max_idx])

        if self.mode in ("min", "both"):
            min_val = path[0]
            min_idx = 0
            for i, s in enumerate(path):
                if s < min_val:
                    min_val = s
                    min_idx = i
            state["min"] = float(min_val)
            if self.track_time:
                state["min_time"] = float(timegrid[min_idx])
