import numpy as np
import QuantLib as ql

from .base import Model


class HestonModel(Model):
    def __init__(self, s0_handle, riskFreeTS, dividendTS, v0, kappa, theta, sigma, rho):
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)

        self._s0 = s0_handle
        self._r = riskFreeTS
        self._q = dividendTS

        self._process = ql.HestonProcess(
            self._r, self._q, self._s0, self.v0, self.kappa, self.theta, self.sigma, self.rho
        )
        self._ql_model = ql.HestonModel(self._process)

    def s0(self):
        return float(self._s0.value())

    def discount(self, t: float):
        return float(self._r.discount(float(t)))

    def path_sampler(self):
        process = self._process

        def sampler(n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
            seed = int(randnumbergenerator.integers(2**31 - 1))
            t_grid = ql.TimeGrid(list(map(float, timegrid)))
            n_steps = len(t_grid) - 1
            dim = process.factors() * n_steps

            ursg = ql.UniformRandomSequenceGenerator(dim, ql.UniformRandomGenerator(seed))
            grsg = ql.GaussianRandomSequenceGenerator(ursg)
            pg = ql.GaussianMultiPathGenerator(process, t_grid, grsg, False)

            s = np.empty((n_paths, len(t_grid)))
            for i in range(n_paths):
                path = pg.next().value()
                spot_path = path[0]
                s[i, :] = np.fromiter((spot_path[j] for j in range(len(t_grid))), dtype=float)
            return s

        return sampler
