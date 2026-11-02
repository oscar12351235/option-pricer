"""Jump components for jump-diffusion samplers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Jump(Protocol):
    """Multiplicative jump component of a jump-diffusion.

    ``compensator`` is the expected relative jump size ``k = E[e^J] - 1``. A
    sampler subtracts ``lambda * k`` from the drift so that the discounted price
    remains a martingale. ``step_multiplier`` returns the product of the jump
    factors ``e^J`` over one time step for each of ``n`` paths.
    """

    def compensator(self) -> float: ...

    def step_multiplier(
        self,
        n: int,
        dt: float,
        randnumbergenerator: np.random.Generator,
        z: np.ndarray | None = None,
    ) -> np.ndarray: ...


class NoJump:
    """No jumps: pure diffusion."""

    def compensator(self) -> float:
        return 0.0

    def step_multiplier(
        self,
        n: int,
        dt: float,
        randnumbergenerator: np.random.Generator | None,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.ones(n)


class LognormalJumps:
    """Compound-Poisson jumps with normally distributed log sizes.

    Jumps arrive at rate ``lam`` per year and each log jump size is
    ``J ~ N(mu, sigma^2)``, so ``k = exp(mu + sigma^2 / 2) - 1``. This is the
    Merton (1976) / Bates (1996) convention and corresponds to QuantLib's
    ``BatesProcess(lambda=lam, nu=mu, delta=sigma)``.
    """

    def __init__(self, lam: float, mu: float, sigma: float) -> None:
        self.lam = float(lam)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def compensator(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma * self.sigma) - 1.0)

    def step_multiplier(
        self,
        n: int,
        dt: float,
        randnumbergenerator: np.random.Generator,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw the jump factor for ``n`` paths over a step of length ``dt``.

        ``N ~ Poisson(lam * dt)`` jumps occur on each path; given ``N``, the sum
        of the log jumps is ``N * mu + sigma * sqrt(N) * Z``. ``z`` optionally
        supplies the normals ``Z``, either one per path or one per path that
        jumps.
        """
        N = randnumbergenerator.poisson(self.lam * dt, size=n)
        eta = np.ones(n, dtype=float)
        idx = np.nonzero(N)[0]
        if idx.size > 0:
            if z is None:
                z_sel = randnumbergenerator.standard_normal(idx.size)
            else:
                z_arr = np.asarray(z)
                if z_arr.shape[0] == n:
                    z_sel = z_arr[idx]
                elif z_arr.shape[0] == idx.size:
                    z_sel = z_arr
                else:
                    raise ValueError("z must have length n or number of jumps")
            eta[idx] = np.exp(self.mu * N[idx] + self.sigma * np.sqrt(N[idx]) * z_sel)
        return eta
