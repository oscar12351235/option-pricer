from __future__ import annotations

import math

from .base import Model, PathSampler


class BatesModel(Model):
    """Flat-rate model around a NumPy/Numba path sampler.

    Pairs ``sampler`` with continuous discounting at the constant rate ``r``.
    It is used with :class:`~option_pricer.models.NumbaHestonSampler` (Heston,
    or Bates when jumps are given) and with
    :class:`~option_pricer.models.VectorizedJumpGBMSampler`. The sampler's
    ``supports_crn`` / ``supports_antithetic`` capabilities are forwarded to
    the engine.
    """

    def __init__(self, sampler: PathSampler, S0: float, r: float, q: float) -> None:
        self._sampler = sampler
        self._S0 = float(S0)
        self._r = float(r)
        self._q = float(q)
        self.supports_crn = getattr(sampler, "supports_crn", False)
        self.supports_antithetic = getattr(sampler, "supports_antithetic", False)

    def s0(self) -> float:
        return self._S0

    def discount(self, t: float) -> float:
        return math.exp(-self._r * float(t))

    def path_sampler(self) -> PathSampler:
        return self._sampler
