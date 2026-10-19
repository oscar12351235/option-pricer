import numpy as np

from .base import Model


class BatesModel(Model):
    def __init__(self, sampler, S0, r, q):
        self._sampler = sampler
        self._S0 = float(S0)
        self._r = float(r)
        self._q = float(q)
        self.supports_crn = getattr(sampler, "supports_crn", False)
        self.supports_antithetic = getattr(sampler, "supports_antithetic", False)

    def s0(self):
        return self._S0

    def discount(self, t):
        return float(np.exp(-self._r * float(t)))

    def path_sampler(self):
        return self._sampler
