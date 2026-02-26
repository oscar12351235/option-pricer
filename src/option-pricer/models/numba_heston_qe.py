from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from numba import njit, prange


@njit
def heston_qe_step_scalar(
    s: float,
    v: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    q: float,
    dt: float,
    zv: float,
    zind: float,
    v_floor: float,
) -> Tuple[float, float]:
    vt = v if v > v_floor else v_floor
    exp_kdt = math.exp(-kappa * dt)

    s2 = (
        vt * sigma * sigma * exp_kdt / kappa * (1.0 - exp_kdt)
        + theta * sigma * sigma / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
    )
    m = theta + (vt - theta) * exp_kdt
    m2 = m * m
    psi = 1e9 if m <= 0.0 else s2 / m2

    if psi < 1.5:
        two_over_psi = 2.0 / psi
        b2 = (two_over_psi - 1.0) + math.sqrt(two_over_psi) * math.sqrt(two_over_psi - 1.0)
        a = m / (1.0 + b2)
        v_next = a * (math.sqrt(b2) + zv) ** 2
    else:
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        U = np.random.random()
        if U < p:
            v_next = 0.0
        else:
            v_next = math.log((1.0 - p) / (1.0 - U)) / beta

    zs = rho * zv + math.sqrt(max(1.0 - rho * rho, 0.0)) * zind
    v_used = v_next if v_next > v_floor else v_floor
    vol = math.sqrt(v_used)

    drift = (r - q - 0.5 * vol * vol) * dt
    diff = vol * math.sqrt(dt) * zs
    s_next = s * math.exp(drift + diff)

    return s_next, v_next


@njit(parallel=True)
def simulate_heston_paths_numba_scalar(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    q: float,
    timegrid: np.ndarray,
    v_floor: float,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    np.random.seed(seed)

    n_times = timegrid.shape[0]
    n_steps = n_times - 1
    paths = np.empty((n_paths, n_times), dtype=np.float64)

    for i in prange(n_paths):
        s = S0
        v = v0
        paths[i, 0] = S0

        for t in range(n_steps):
            dt = timegrid[t + 1] - timegrid[t]
            zv = np.random.normal()
            zind = np.random.normal()
            s, v = heston_qe_step_scalar(
                s, v, kappa, theta, sigma, rho, r, q, dt, zv, zind, v_floor
            )
            paths[i, t + 1] = s

    return paths


class NumbaHestonSampler:
    def __init__(self, S0, v0, kappa, theta, sigma, rho, r, q, v_floor=1e-12):
        self.S0 = float(S0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.r = float(r)
        self.q = float(q)
        self.v_floor = float(v_floor)

        self.supports_crn = False
        self.supports_antithetic = False

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
        timegrid = np.asarray(timegrid, dtype=np.float64)
        seed = int(randnumbergenerator.integers(2**31 - 1))
        return simulate_heston_paths_numba_scalar(
            self.S0,
            self.v0,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.r,
            self.q,
            timegrid,
            self.v_floor,
            int(n_paths),
            seed,
        )
