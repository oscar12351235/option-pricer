from typing import Protocol

import numpy as np


class Jump(Protocol):
    def compensator(self) -> float: ...

    def step_multiplier(
        self, n: int, dt: float, randnumbergenerator, z: np.ndarray | None = None
    ): ...


class NoJump:
    def compensator(self) -> float:
        return 0.0

    def step_multiplier(self, n, dt, randnumbergenerator, z=None):
        return np.ones(n)


class LognormalJumps:
    def __init__(self, lam: float, mu: float, sigma: float):
        self.lam = float(lam)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def compensator(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma * self.sigma) - 1.0)

    def step_multiplier(self, n, dt, randnumbergenerator, z=None):
        N = randnumbergenerator.poisson(self.lam * dt, size=n)
        eta = np.ones(n, dtype=float)
        idx = np.nonzero(N)[0]
        if idx.size > 0:
            if z is None:
                z_eff = randnumbergenerator.standard_normal(idx.size)
            else:
                z_arr = np.asarray(z)
                if z_arr.shape[0] == n:
                    z_eff = z_arr[idx]
                elif z_arr.shape[0] == idx.size:
                    z_eff = z_arr
                else:
                    raise ValueError(
                        "z must have length n "
                        f"({n}) or number of jumps ({idx.size}); got {z_arr.shape[0]}"
                    )
            eta[idx] = np.exp(self.mu * N[idx] + self.sigma * np.sqrt(N[idx]) * z_eff)
        return eta
