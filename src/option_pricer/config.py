from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloConfig:
    n_paths: int
    timegrid: np.ndarray
    seed: int | None = None
    use_crn: bool = False
    antithetic: bool = False
