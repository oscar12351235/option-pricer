"""Exercise policies: which indices of the time grid allow exercise."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Exercise(Protocol):
    """Decides whether the option may be exercised at ``timegrid[timeindex]``."""

    def exercise_time(self, timeindex: int, timegrid: np.ndarray) -> bool: ...


class European:
    """Exercise at maturity only."""

    def exercise_time(self, timeindex: int, timegrid: np.ndarray) -> bool:
        return timeindex == len(timegrid) - 1


class American:
    """Early exercise at every interior grid date.

    The payoff at maturity is always available to the Longstaff-Schwartz
    pricer, so it is not flagged here. On a discrete grid this is a Bermudan
    approximation of American exercise that converges as the grid is refined.
    """

    def exercise_time(self, timeindex: int, timegrid: np.ndarray) -> bool:
        return 0 < timeindex < len(timegrid) - 1
