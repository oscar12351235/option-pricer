from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MonteCarloConfig:
    n_paths: int
    timegrid: np.ndarray
    seed: Optional[int] = None
    use_crn: bool = False
    antithetic: bool = False
