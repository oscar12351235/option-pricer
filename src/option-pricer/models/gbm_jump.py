from __future__ import annotations

import numpy as np

from .jumps import Jump


class VectorizedJumpGBMSampler:
    """
    Vectorized jump-diffusion GBM sampler.

    The sampler is compatible with `MonteCarloEngine` and supports CRN/antithetic
    through the `z` tensor contract used by numpy-based models.
    """

    supports_crn = True
    supports_antithetic = True

    def __init__(self, S0: float, r: float, q: float, sigma: float, jump: Jump):
        self.S0 = float(S0)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.jump = jump

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
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
            z_diff = z[step, :, 0]
            z_jump = z[step, :, 1]

            drift = (
                self.r - self.q - 0.5 * self.sigma * self.sigma - jump_intensity * jump_compensator
            ) * dt
            diffusion = self.sigma * np.sqrt(dt) * z_diff
            jump_multiplier = self.jump.step_multiplier(
                n=n_paths,
                dt=dt,
                randnumbergenerator=randnumbergenerator,
                z=z_jump,
            )

            paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion) * jump_multiplier

        return paths
