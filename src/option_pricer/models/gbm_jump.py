from __future__ import annotations

import numpy as np

from .jumps import Jump


class VectorizedJumpGBMSampler:
    """Geometric Brownian motion with optional jumps (Merton 1976), vectorised over paths.

    Simulates ``dS/S = (r - q - lambda * k) dt + sigma dW + (e^J - 1) dN`` exactly
    on the grid. Factor 0 of ``z`` drives the diffusion and factor 1 the jump
    sizes; jump counts are drawn from ``randnumbergenerator``. With
    :class:`~option_pricer.models.NoJump` this is Black-Scholes.
    """

    supports_crn = True
    supports_antithetic = True

    def __init__(self, S0: float, r: float, q: float, sigma: float, jump: Jump) -> None:
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.jump = jump

    def __call__(
        self,
        n_paths: int,
        timegrid: np.ndarray,
        randnumbergenerator: np.random.Generator,
        randnumber_key: str | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        n_paths = int(n_paths)
        timegrid = np.asarray(timegrid, dtype=float)
        n_steps = len(timegrid) - 1

        if z is None:
            z = randnumbergenerator.standard_normal(size=(n_steps, n_paths, 2))

        paths = np.empty((n_paths, len(timegrid)), dtype=float)
        paths[:, 0] = self.S0

        jump_intensity = float(getattr(self.jump, "lam", 0.0))
        jump_compensator = float(self.jump.compensator())

        for step in range(n_steps):
            dt = float(timegrid[step + 1] - timegrid[step])
            diffusion_z = z[step, :, 0]
            jump_z = z[step, :, 1]

            drift = (
                self.r - self.q - jump_intensity * jump_compensator - 0.5 * self.sigma * self.sigma
            ) * dt
            diffusion = self.sigma * np.sqrt(dt) * diffusion_z
            jump_multiplier = self.jump.step_multiplier(
                n_paths,
                dt,
                randnumbergenerator,
                z=jump_z,
            )

            paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion) * jump_multiplier

        return paths
