from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloConfig:
    """Simulation settings for :class:`~option_pricer.engine.MonteCarloEngine`.

    Attributes:
        n_paths: Number of simulated paths (must be even with ``antithetic``).
        timegrid: Increasing simulation times in years, starting at 0.
        seed: Seed of the engine's ``numpy.random.Generator``.
        use_crn: Draw the normal tensor once and reuse it for every simulation
            made by the engine (common random numbers).
        antithetic: Pair every normal draw ``z`` with ``-z``.
    """

    n_paths: int
    timegrid: np.ndarray
    seed: int | None = None
    use_crn: bool = False
    antithetic: bool = False
