"""Numba-accelerated Heston / Bates path simulation with the Andersen QE scheme.

The variance is advanced with the Quadratic-Exponential (QE) scheme of
Andersen (2008), "Simple and efficient simulation of the Heston stochastic
volatility model", J. Comput. Finance 11(3). The log-asset uses Andersen's
central discretisation of the integrated variance (gamma_1 = gamma_2 = 1/2)
together with his martingale correction, so that ``E[S_{t+dt} | S_t, V_t]``
equals ``S_t * exp((r - q) dt)`` on every step.

Bates (1996) dynamics are obtained by adding compound-Poisson lognormal jumps
to the log-asset, compensated so that the discounted price stays a martingale:

    dS/S = (r - q - lambda * k) dt + sqrt(V) dW_S + (e^J - 1) dN,
    J ~ N(mu_J, sigma_J^2),  k = exp(mu_J + sigma_J^2 / 2) - 1.

All randomness is taken from a tensor of standard normals ``z`` with shape
``(n_steps, n_paths, n_factors)``. Uniform variates (QE exponential branch,
Poisson counts) are obtained as ``Phi(z)``, which keeps every path a
deterministic function of its normals. This is what lets the engine apply
common random numbers and antithetic variates to this sampler, and makes the
output independent of the number of Numba threads.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .jumps import Jump, LognormalJumps, NoJump

PSI_CRITICAL = 1.5
"""Switching threshold between the quadratic and exponential QE branches."""

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


@njit(cache=True)
def _std_normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x * _INV_SQRT2)


@njit(cache=True)
def qe_variance_step(
    v: float,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
    z_v: float,
) -> tuple[float, float, float, float, bool]:
    """Advance the CIR variance over ``dt`` with Andersen's QE scheme.

    Returns ``(v_next, a, b2_or_p, beta, quadratic)``. The middle values are
    the branch parameters, needed by the martingale correction of the asset
    step: ``(a, b^2)`` in the quadratic branch, ``(p, beta)`` in the
    exponential branch.
    """
    exp_kdt = math.exp(-kappa * dt)
    m = theta + (v - theta) * exp_kdt
    s2 = v * sigma * sigma * exp_kdt / kappa * (1.0 - exp_kdt) + theta * sigma * sigma / (
        2.0 * kappa
    ) * (1.0 - exp_kdt) * (1.0 - exp_kdt)
    psi = s2 / (m * m)

    if psi <= PSI_CRITICAL:
        inv_psi2 = 2.0 / psi
        b2 = inv_psi2 - 1.0 + math.sqrt(inv_psi2) * math.sqrt(inv_psi2 - 1.0)
        a = m / (1.0 + b2)
        root = math.sqrt(b2) + z_v
        return a * root * root, a, b2, 0.0, True

    p = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p) / m
    u = _std_normal_cdf(z_v)
    v_next = 0.0 if u <= p else math.log((1.0 - p) / (1.0 - u)) / beta
    return v_next, 0.0, p, beta, False


@njit(cache=True)
def _poisson_from_uniform(u: float, mean: float) -> int:
    """Inverse-CDF Poisson draw; the loop is short because ``mean = lambda*dt`` is small."""
    n = 0
    pmf = math.exp(-mean)
    cdf = pmf
    while u > cdf and pmf > 0.0:
        n += 1
        pmf *= mean / n
        cdf += pmf
    return n


@njit(cache=True, parallel=True)
def simulate_bates_qe(
    s0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    q: float,
    jump_lambda: float,
    jump_mu: float,
    jump_sigma: float,
    timegrid: np.ndarray,
    z: np.ndarray,
    store_variance: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston (``jump_lambda == 0``) or Bates paths on ``timegrid``.

    ``z`` has shape ``(n_steps, n_paths, n_factors)``: factor 0 drives the
    variance, factor 1 the asset (independent of factor 0; the correlation
    enters through the integrated-variance terms), and when jumps are enabled
    factors 2 and 3 drive the Poisson counts and the jump sizes.

    Returns ``(spot_paths, variance_paths)`` with shape ``(n_paths, n_times)``;
    ``variance_paths`` has shape ``(0, 0)`` unless ``store_variance`` is true.
    """
    n_times = timegrid.shape[0]
    n_paths = z.shape[1]
    has_jumps = jump_lambda > 0.0
    jump_comp = jump_lambda * (math.exp(jump_mu + 0.5 * jump_sigma * jump_sigma) - 1.0)

    spots = np.empty((n_paths, n_times), dtype=np.float64)
    if store_variance:
        variances = np.empty((n_paths, n_times), dtype=np.float64)
    else:
        variances = np.empty((0, 0), dtype=np.float64)

    for i in prange(n_paths):
        log_s = math.log(s0)
        v = v0
        spots[i, 0] = s0
        if store_variance:
            variances[i, 0] = v0

        for t in range(n_times - 1):
            dt = timegrid[t + 1] - timegrid[t]
            v_next, a, b2_or_p, beta, quadratic = qe_variance_step(
                v, dt, kappa, theta, sigma, z[t, i, 0]
            )

            # Andersen (2008), eq. (33) with gamma_1 = gamma_2 = 1/2.
            k1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
            k2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
            k3 = 0.5 * dt * (1.0 - rho * rho)
            k4 = k3
            big_a = k2 + 0.5 * k4

            # Martingale correction K0* (Andersen 2008, Section 4.3.2); falls
            # back to the uncorrected K0 when the moment-generating function
            # of V_{t+dt} does not exist at big_a.
            if quadratic and big_a < 0.5 / a:
                k0 = (
                    -big_a * b2_or_p * a / (1.0 - 2.0 * big_a * a)
                    + 0.5 * math.log(1.0 - 2.0 * big_a * a)
                    - (k1 + 0.5 * k3) * v
                )
            elif (not quadratic) and big_a < beta:
                p = b2_or_p
                k0 = -math.log(p + beta * (1.0 - p) / (beta - big_a)) - (k1 + 0.5 * k3) * v
            else:
                k0 = -rho * kappa * theta * dt / sigma

            log_s += (
                (r - q - jump_comp) * dt
                + k0
                + k1 * v
                + k2 * v_next
                + math.sqrt(max(k3 * v + k4 * v_next, 0.0)) * z[t, i, 1]
            )

            if has_jumps:
                n_jumps = _poisson_from_uniform(_std_normal_cdf(z[t, i, 2]), jump_lambda * dt)
                if n_jumps > 0:
                    log_s += n_jumps * jump_mu + jump_sigma * math.sqrt(n_jumps) * z[t, i, 3]

            v = v_next
            spots[i, t + 1] = math.exp(log_s)
            if store_variance:
                variances[i, t + 1] = v

    return spots, variances


class NumbaHestonSampler:
    """Path sampler for the Heston model, or the Bates model when given jumps.

    Parameters follow the usual Heston notation: ``v0`` initial variance,
    ``kappa`` mean-reversion speed, ``theta`` long-run variance, ``sigma``
    volatility of variance and ``rho`` the spot/variance correlation. ``r`` and
    ``q`` are continuously compounded rate and dividend yield. Passing
    ``jump=LognormalJumps(lam, mu, sigma)`` adds Bates jumps; ``None`` or
    :class:`NoJump` gives pure Heston.

    The sampler consumes normals supplied by the engine (``z``), so it
    supports common random numbers and antithetic variates.
    """

    supports_crn = True
    supports_antithetic = True

    def __init__(
        self,
        S0: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        r: float,
        q: float,
        jump: Jump | None = None,
    ) -> None:
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")
        if v0 < 0.0:
            raise ValueError("v0 must be non-negative")
        if kappa <= 0.0 or theta <= 0.0 or sigma <= 0.0:
            raise ValueError("kappa, theta and sigma must be positive")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")

        self.S0 = float(S0)
        self.v0 = float(v0)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.r = float(r)
        self.q = float(q)

        if jump is None or isinstance(jump, NoJump):
            self.jump_lambda, self.jump_mu, self.jump_sigma = 0.0, 0.0, 0.0
        elif isinstance(jump, LognormalJumps):
            if jump.lam < 0.0 or jump.sigma < 0.0:
                raise ValueError("jump intensity and jump volatility must be non-negative")
            self.jump_lambda, self.jump_mu, self.jump_sigma = jump.lam, jump.mu, jump.sigma
        else:
            raise TypeError("jump must be None, NoJump or LognormalJumps")

    @property
    def n_factors(self) -> int:
        """Number of normals consumed per path and time step."""
        return 4 if self.jump_lambda > 0.0 else 2

    def simulate(
        self,
        n_paths: int,
        timegrid: np.ndarray,
        randnumbergenerator: np.random.Generator,
        z: np.ndarray | None = None,
        store_variance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(spot_paths, variance_paths)``, each of shape ``(n_paths, n_times)``."""
        timegrid = np.asarray(timegrid, dtype=np.float64)
        n_steps = len(timegrid) - 1
        expected = (n_steps, int(n_paths), self.n_factors)
        if z is None:
            z = randnumbergenerator.standard_normal(size=expected)
        z = np.ascontiguousarray(z, dtype=np.float64)
        if z.shape != expected:
            raise ValueError(f"z must have shape {expected}, got {z.shape}")

        return simulate_bates_qe(
            self.S0,
            self.v0,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.r,
            self.q,
            self.jump_lambda,
            self.jump_mu,
            self.jump_sigma,
            timegrid,
            z,
            store_variance,
        )

    def __call__(
        self,
        n_paths: int,
        timegrid: np.ndarray,
        randnumbergenerator: np.random.Generator,
        randnumber_key: str | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        spots, _ = self.simulate(n_paths, timegrid, randnumbergenerator, z, store_variance=False)
        return spots
