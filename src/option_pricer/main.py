import QuantLib as ql
import math
import random
from datetime import date, timedelta
from abc import ABC, abstractmethod
from typing import (
    Literal,
    Sequence,
    Protocol,
    FrozenSet,
    Optional,
    Mapping,
    MutableMapping,
    Any,
    Dict,
    Tuple,
)
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
from numba import njit, prange
from line_profiler import profile
from memory_profiler import profile


class Payoff(ABC):

    def __init__(self):
        self.requires: FrozenSet[str] = frozenset()

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> float:
        pass


###############
### PAYOFFS ###
###############
class Vanilla(Payoff):
    """single-key payoff, can represent vanilla if key is set to terminal, asian if key is set to average"""

    def __init__(self, key: str, k: float, side: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        super().__init__()
        self.key = key
        self.k = k
        self.side = side
        self.requires = frozenset({self.key})

    def __call__(self, state):

        x = state[self.key]

        if self.side == "call":
            return max(x - self.k, 0.0)
        if self.side == "put":
            return max(self.k - x, 0.0)


class Digital(Payoff):
    """generic digital with side either call (>) or put (<) and variant either "cash" or "asset",
    key is which state top read ("terminal", "average", etc...), and cash is the cash amount
    in the cash variant. Asian Fixed-Digital is covered by this class if x is set to key = average
    """

    def __init__(self, key: str, k: float, side: str, cash: float, variant: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        if variant not in ("cash", "asset"):
            raise ValueError("variant should be 'cash' or 'asset'")
        super().__init__()
        self.key = key
        self.k = k
        self.side = side
        self.variant = variant
        self.cash = cash
        self.requires = frozenset({self.key})

    def __call__(self, state):
        x = float(state[self.key])
        # x a generic placeholder to make the system modular depending on the key

        if self.side == "call":
            if x > self.k:
                if self.variant == "cash":
                    return self.cash
                else:
                    return x
            else:
                return 0

        if self.side == "put":
            if x < self.k:
                if self.variant == "cash":
                    return self.cash
                else:
                    return x
            else:
                return 0


class AsianFloatingVanilla(Payoff):
    """fixed-strike asian compares average price vs fixed k whereas
    floating strike compares final price to average price as strike k"""

    def __init__(self, key1: str, key2: str, side: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.side = side
        self.requires = frozenset({key1, key2})

    def __call__(self, state):
        x = float(state[self.key1])
        k = float(state[self.key2])

        if self.side == "call":
            return max(x - k, 0.0)

        if self.side == "put":
            return max(k - x, 0.0)


# class AsianFloatingDigital(Payoff):


class KnockOutAnyN(Payoff):
    """N-imbricated knockout (knockout, double knockout, triple knockout, etc...)
    which returns 0 if any single barrier is hit"""

    def __init__(self, inner: Payoff, keys):
        super().__init__()
        self.inner = inner
        self.keys = keys
        unique_keys = list(dict.fromkeys(keys))
        self.requires = inner.requires | frozenset(unique_keys)

    def __call__(self, state):
        for k in self.keys:
            if state.get(k, False):
                return 0.0

        return self.inner(state)


class KnockInAllN(Payoff):
    """N-imbricated knock-in (knock-in, double knock-in, triple knock-in, etc...)
    which pays out only if all barriers are hit"""

    def __init__(self, inner: Payoff, keys):
        super().__init__()
        self.inner = inner
        self.keys = keys
        unique_keys = list(dict.fromkeys(keys))
        self.requires = inner.requires | frozenset(unique_keys)

    def __call__(self, state):
        for k in self.keys:
            if not state.get(k, False):
                return 0.0

        return self.inner(state)


class KnockInKnockOut(Payoff):
    """standard KIKO, pays out iff KI happened and KO never happened"""

    def __init__(self, inner: Payoff, ki_key, ko_key):
        if ki_key == ko_key:
            raise ValueError(" 'ki_key' and 'ko_key' should be different ")
        super().__init__()
        self.inner = inner
        self.ki_key = ki_key
        self.ko_key = ko_key
        self.requires = inner.requires | frozenset({ki_key, ko_key})

    def __call__(self, state):
        if not state.get(self.ki_key, False):
            return 0.0
        else:
            if state.get(self.ko_key, False):
                return 0.0
            else:
                return self.inner(state)


#####################
### PATH FEATURES ###
#####################
class Feature(ABC):
    @property
    @abstractmethod
    def provides(self):
        pass

    @abstractmethod
    def apply(self, path, state: dict) -> None:
        pass


class AsianFeature(Feature):
    @property
    def provides(self):
        return frozenset({"average"})

    def apply(self, path, state):
        # path is a sequence of simulated underlying prices along the time grid for one Monte Carlo path.
        state["average"] = sum(path) / len(path)


class BarrierFeature(Feature):
    def __init__(self, name: str, level: float, direction: str, strict: bool, track_time: bool):
        if direction not in ("up", "down"):
            raise ValueError("Direction should be 'up' or 'down'")
        self.name = name
        self.level = level
        self.direction = direction
        self.strict = strict
        self.track_time = track_time

    @property
    def provides(self):
        keys = {f"{self.name}_flag"}
        if self.track_time:
            keys.add(f"{self.name}_time")
        return frozenset(keys)

    def apply(self, path, state):
        hit_time_id = None
        if self.direction == "up":
            if self.strict:
                for i, s in enumerate(path):
                    if float(s) > self.level:
                        hit_time_id = i
                        break
            else:
                for i, s in enumerate(path):
                    if float(s) >= self.level:
                        hit_time_id = i
                        break
        elif self.direction == "down":
            if self.strict:
                for i, s in enumerate(path):
                    if float(s) < self.level:
                        hit_time_id = i
                        break
            else:
                for i, s in enumerate(path):
                    if float(s) <= self.level:
                        hit_time_id = i
                        break

        if hit_time_id is not None:
            state[f"{self.name}_flag"] = True
        else:
            state[f"{self.name}_flag"] = False

        timegrid = state.get("t_grid")

        if self.track_time:
            if hit_time_id is None:
                state[f"{self.name}_time"] = math.inf
            else:
                if len(timegrid) != len(path):
                    raise ValueError("state['t_grid'] length must equal len(path)")
                if isinstance(timegrid, (list, tuple, np.ndarray)) is False:
                    raise ValueError("state['t_grid'] must be a list or tuple of times")

                state[f"{self.name}_time"] = timegrid[hit_time_id]


class LookbackFeature(Feature):
    def __init__(self, mode: str, track_time: bool):
        if mode not in ("max", "min", "both"):
            raise ValueError("mode should be 'max', 'min' or 'both'")
        self.mode = mode
        self.track_time = track_time

    @property
    def provides(self):
        keys = set()
        if self.mode == "max":
            keys.add("max")
            if self.track_time:
                keys.add("max_time")
        elif self.mode == "min":
            keys.add("min")
            if self.track_time:
                keys.add("min_time")
        elif self.mode == "both":
            keys.add("min")
            keys.add("max")
            if self.track_time:
                keys.add("min_time")
                keys.add("max_time")
        return frozenset(keys)

    def apply(self, path, state):
        timegrid = state.get("t_grid")
        if self.mode in ["max", "both"]:
            max_val = path[0]
            max_idx = 0
            for i, s in enumerate(path):
                if s > max_val:
                    max_val = s
                    max_idx = i
            state["max"] = max_val
            if self.track_time:
                if len(timegrid) != len(path):
                    raise ValueError("state['t_grid'] length must equal len(path)")
                if isinstance(timegrid, (list, tuple)) is False:
                    raise ValueError("state['t_grid'] must be a list or tuple of times")
                state["max_time"] = timegrid[max_idx]

        if self.mode in ["min", "both"]:
            min_val = path[0]
            min_idx = 0
            for i, s in enumerate(path):
                if s < min_val:
                    min_val = s
                    min_idx = i
            state["min"] = min_val
            if self.track_time:
                if len(timegrid) != len(path):
                    raise ValueError("state['t_grid'] length must equal len(path)")
                if isinstance(timegrid, (list, tuple)) is False:
                    raise ValueError("state['t_grid'] must be a list or tuple of times")
                state["min_time"] = timegrid[min_idx]


################
### EXERCISE ###
################
class Exercise(Protocol):
    def exercise_time(self, timeindex: int, timegrid):
        pass


class European:
    def exercise_time(self, timeindex, timegrid):
        return timeindex == len(timegrid) - 1


class American:
    def exercise_time(self, timeindex, timegrid):
        return 0 < timeindex < len(timegrid) - 1


#############
### MODEL ###
#############


class PathSampler(Protocol):
    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key, z):
        pass


class Model(ABC):
    @abstractmethod
    def s0(self) -> float:
        pass

    @abstractmethod
    def discount(self, t: float) -> float:
        pass

    @abstractmethod
    def path_sampler(self) -> PathSampler:
        pass


class HestonModel(Model):
    """We have s0_handle as initial condition for level S0, riskFreeTS as the risk-free term structure (provides r(t)),
    dividendTS as the dividend term structure (the carry, provides the q(t)), v0 as initial variance (vol^2),
    theta as the long-run variance θ, kappa as the mean-reversion speed κ (i.e. how fast v reverts to θ), sigma as the
    volatility of volatility, rho as the correlation between Ws (BM of asset) and Wv (BM of variance) (correlation is assumed to be
    constant)."""

    def __init__(self, s0_handle, riskFreeTS, dividendTS, v0, kappa, theta, sigma, rho):
        self.riskFreeTS = riskFreeTS
        self.dividendTS = dividendTS
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self._s0 = s0_handle
        self._r = riskFreeTS
        self._q = dividendTS
        self._process = ql.HestonProcess(
            self._r, self._q, self._s0, self.v0, self.kappa, self.theta, self.sigma, self.rho
        )
        self._ql_model = ql.HestonModel(self._process)

        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(" correlation rho must take values in range [-1; 1] ")

        if self.v0 < 0 or self.theta < 0 or self.sigma < 0 or self.kappa <= 0:
            raise ValueError(" v0, theta, sigma and kappa must be positive ")

        if self.kappa * self.theta < 1 / 2 * self.sigma * self.sigma:
            print(" Warning, Feller condition breached ")

    def s0(self):
        return self._s0.value()

    def discount(self, t: float):
        return self._r.discount(t)

    def path_sampler(self):
        """path sampler configures how to simulate paths (which process, random generator, etc.). Seed creates an
        independent seed so that each call gives a different seed to ensure independent simulations if you repeat.
        2 ** 31 - 1 just ensures the number is within the 32bit seed limit of Quantlib. uniformrandnumbergenerator
        generates sequences of uniform [0,1] random numbers and gaussianrandnumbergenerator transforms those sequences
        into standard normal deviates. We loop over for n paths."""
        process = self._process

        def sampler(n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
            seed = int(randnumbergenerator.integers(2**31 - 1))
            t_grid = ql.TimeGrid(list(map(float, timegrid)))
            n_steps = len(t_grid) - 1
            factors = process.factors()
            dim = factors * n_steps
            uniformrandnumbergenerator = ql.UniformRandomSequenceGenerator(
                dim, ql.UniformRandomGenerator(seed)
            )
            gaussianrandnumbergenerator = ql.GaussianRandomSequenceGenerator(
                uniformrandnumbergenerator
            )
            pathgenerator = ql.GaussianMultiPathGenerator(
                process, t_grid, gaussianrandnumbergenerator, False
            )

            s = np.empty((n_paths, len(t_grid)))
            for i in range(n_paths):
                path = pathgenerator.next().value()
                spot_path = path[0]
                for j in range(len(t_grid)):
                    s[i, j] = spot_path[j]
            return s

        return sampler


### For Bates Jump-Diffusion model: Heston with Jump = NoJump and Bates with Jump = LognormalJumps ###
class Jump(Protocol):
    """The compensator measures the relative jump size so that it can later be compensated to remain risk-neutral.
    Indeed, if on average jumps make prices rise by k each time they happen, and they occur at rate λ, then without
    correction the expected growth rate would be inflated by λk. The step multiplier simulates n paths (scenarios) over
    a small time-step dt."""

    def compensator(self) -> float:
        pass

    def step_multiplier(
        self, n: int, dt: float, randnumbergenerator, z: Optional[np.ndarray] = None
    ):
        pass


class NoJump:
    def compensator(self) -> float:
        return 0.0

    def step_multiplier(self, n, dt, randnumbergenerator, z):
        return np.ones(n)


class LognormalJumps:
    """This part models the third term of the asset price process corresponding to the Jump term. lam is for lambda,
    representing the jump intensity (expected number of jumps per unit of time), mu represents the mean of the size of
    jumps, and sigma represents the volatility of the jump sizes (how big and how erratic jumps can be). The compensator
    measures the relative jump size so that it can later be compensated to remain risk-neutral. Indeed, if on average
    jumps make prices rise by k each time they happen, and they occur at rate λ, then without correction the expected
    growth rate would be inflated by λk. The step multiplier simulates n paths (scenarios) over a small time-step dt and
    gives the number of jumps in each path (through N). eta (as the greek letter not expected time of arrival) is the
    multiplicative factor Jt for each path. It is initialized as 1, meaning no jumps. idx is the list of the index of
    the non-zero elements of N (so the index of the paths with at least one jump in the array of n simulated scenarios).
    idx informs where (which paths) there are jumps, eta give the magnitude of each those jumps."""

    def __init__(self, lam: float, mu: float, sigma: float):
        self.lam = float(lam)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def compensator(self) -> float:
        return np.exp(self.mu + 0.5 * self.sigma * self.sigma) - 1.0

    def step_multiplier(self, n, dt, randnumbergenerator, z):
        N = randnumbergenerator.poisson(self.lam * dt, size=n)
        eta = np.ones(n, dtype=float)
        idx = np.nonzero(N)[0]
        if len(idx) > 0:
            z = z if z is not None else randnumbergenerator.standard_normal(idx.size)
            eta[idx] = np.exp(self.mu * N[idx] + self.sigma * np.sqrt(N[idx]) * z)
        return eta


class BatesModel(Model):
    """NumPy QE Bates model with CRN & antithetic support"""

    def __init__(self, sampler, S0, r, q):
        self._sampler = sampler
        self._S0 = float(S0)
        self._r = float(r)
        self._q = float(q)

        # Tell the engine whether we support CRN/antithetic
        self.supports_crn = getattr(sampler, "supports_crn", False)
        self.supports_antithetic = getattr(sampler, "supports_antithetic", False)

    def s0(self):
        return self._S0

    def discount(self, t):
        return np.exp(-self._r * t)

    def path_sampler(self):
        return self._sampler


# ============================
# Numba-accelerated Heston QE
# ============================


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
    """
    Andersen QE step for *one* path / step (pure Heston, no jumps).
    """
    vt = v if v > v_floor else v_floor

    exp_kdt = math.exp(-kappa * dt)

    # conditional variance of v_{t+dt}
    s2 = (
        vt * sigma * sigma * exp_kdt / kappa * (1.0 - exp_kdt)
        + theta * sigma * sigma / (2.0 * kappa) * (1.0 - exp_kdt) ** 2
    )

    # conditional mean of v_{t+dt}
    m = theta + (vt - theta) * exp_kdt
    m2 = m * m

    # psi
    psi = 1e9 if m <= 0.0 else s2 / m2

    # QE branching
    if psi < 1.5:
        # Quadratic
        two_over_psi = 2.0 / psi
        b2 = (two_over_psi - 1.0) + math.sqrt(two_over_psi) * math.sqrt(two_over_psi - 1.0)
        a = m / (1.0 + b2)
        v_next = a * (math.sqrt(b2) + zv) ** 2
    else:
        # Exponential
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        U = np.random.random()
        if U < p:
            v_next = 0.0
        else:
            X = math.log((1.0 - p) / (1.0 - U)) / beta
            v_next = X

    # correlate asset shock
    zs = rho * zv + math.sqrt(max(1.0 - rho * rho, 0.0)) * zind

    # asset step
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
    """
    Parallel simulation of Heston QE paths using scalar step.

    timegrid: shape (n_times,)
    returns: paths shape (n_paths, n_times)
    """
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

            # independent normals
            zv = np.random.normal()
            zind = np.random.normal()

            s, v = heston_qe_step_scalar(
                s,
                v,
                kappa,
                theta,
                sigma,
                rho,
                r,
                q,
                dt,
                zv,
                zind,
                v_floor,
            )

            paths[i, t + 1] = s

    return paths


class NumbaHestonSampler:
    """
    Sampler compatible with MonteCarloEngine, using Numba-compiled QE Heston.
    """

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

        # IMPORTANT: tell the engine NOT to build CRNs/antithetic z-tensors
        self.supports_crn = False
        self.supports_antithetic = False

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key, z):
        """
        Ignore randnumbergenerator and z: RNG is inside Numba kernel.
        """
        timegrid = np.asarray(timegrid, dtype=np.float64)
        # get a seed from the Python RNG for reproducibility
        seed = int(randnumbergenerator.integers(2**31 - 1))

        paths = simulate_heston_paths_numba_scalar(
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
        return paths


########################
### MONTE CARLO ENGINE #
########################


@dataclass
class MonteCarloConfig:
    n_paths: int
    timegrid: np.ndarray
    seed: Optional[int] = None
    use_crn: bool = False
    antithetic: bool = False


def payoff_feature_key(payoff: Payoff, features: Sequence[Feature]) -> None:
    provided = {"terminal"}
    for feat in features:
        provided |= set(feat.provides)

    if not payoff.requires.issubset(provided):
        raise ValueError(
            f"Payoff requires keys {payoff.requires}, "
            f"but only {provided} are available. "
            "Add the missing Features or change the payoff key."
        )


class MonteCarloEngine:
    """
    Generic Monte Carlo engine.

    - European: supports any payoff / features / model.
    - American (and Bermudan via Exercise): supports any payoff / features / model,
      including path-dependent (Asian, lookback, barriers, KIKO, etc.).
    """

    def __init__(
        self,
        model: Model,
        payoff: Payoff,
        features: Sequence[Feature],
        exercise: Exercise,
        config: MonteCarloConfig,
    ):
        self.model = model
        self.payoff = payoff
        self.features = list(features)
        self.exercise = exercise
        self.cfg = config

        payoff_feature_key(payoff, self.features)

        self.timegrid = np.asarray(self.cfg.timegrid, dtype=float)
        if self.timegrid.ndim != 1:
            raise ValueError("timegrid must be 1D")

        if len(self.timegrid) < 2:
            raise ValueError("timegrid must contain at least two time points")

        self.n_paths = int(self.cfg.n_paths)
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if self.cfg.antithetic and (self.n_paths % 2 != 0):
            raise ValueError("n_paths must be even when using antithetic variates")

        self.rng = np.random.default_rng(self.cfg.seed)

        # Precompute discount factors for all grid points
        self._discount_grid = np.array(
            [self.model.discount(float(t)) for t in self.timegrid],
            dtype=float,
        )

        # Precompute CRN tensor Z for NumPy models if requested
        self._z_crn = None
        supports_crn = getattr(self.model, "supports_crn", False)
        supports_antithetic = getattr(self.model, "supports_antithetic", False)

        if self.cfg.use_crn and supports_crn:
            n_steps = len(self.timegrid) - 1
            if self.cfg.antithetic and supports_antithetic:
                n_half = self.n_paths // 2
                z_half = self.rng.standard_normal(size=(n_steps, n_half, 2))
                self._z_crn = np.concatenate([z_half, -z_half], axis=1)
            else:
                self._z_crn = self.rng.standard_normal(size=(n_steps, self.n_paths, 2))

    # ---------- Public API ----------

    def price(self) -> Tuple[float, float]:
        """
        Dispatch to European or American/Bermudan pricer depending on Exercise type.
        """
        if isinstance(self.exercise, European):
            return self._price_european()
        elif isinstance(self.exercise, American):
            return self._price_american_lsm()
        else:
            # Generic Exercise also works (Bermudan) as long as exercise_time() is implemented.
            return self._price_american_lsm()

    # ---------- Path simulation ----------

    def _simulate_paths(self, randnumber_key: str = "base") -> np.ndarray:
        """
        Simulate paths from the model.

        If model is NumPy-based (supports_crn=True), we pass in Z (optionally with antithetic pairs).
        For QuantLib-based models, Z is ignored and only the seed (via rng) matters.
        """
        sampler = self.model.path_sampler()
        supports_crn = getattr(self.model, "supports_crn", False)
        supports_antithetic = getattr(self.model, "supports_antithetic", False)

        z = None
        if supports_crn:
            if self._z_crn is not None:
                # Reuse precomputed Z for CRNs
                z = self._z_crn
            elif self.cfg.antithetic and supports_antithetic:
                # On-the-fly antithetic Z
                n_steps = len(self.timegrid) - 1
                n_half = self.n_paths // 2
                z_half = self.rng.standard_normal(size=(n_steps, n_half, 2))
                z = np.concatenate([z_half, -z_half], axis=1)

        paths = sampler(
            n_paths=self.n_paths,
            timegrid=self.timegrid,
            randnumbergenerator=self.rng,
            randnumber_key=randnumber_key,
            z=z,
        )

        if paths.shape != (self.n_paths, len(self.timegrid)):
            raise ValueError(
                f"Model sampler returned array of shape {paths.shape}, "
                f"expected ({self.n_paths}, {len(self.timegrid)})"
            )
        return paths

    # ---------- European pricing ----------

    def _price_european(self) -> Tuple[float, float]:
        paths = self._simulate_paths(randnumber_key="european")
        df_T = self._discount_grid[-1]

        discounted_payoffs = np.empty(self.n_paths, dtype=float)

        for i in range(self.n_paths):
            path = paths[i, :]
            state: Dict[str, Any] = {
                "t_grid": self.timegrid,
                "terminal": float(path[-1]),
            }

            # Apply all features on the full path
            for feat in self.features:
                feat.apply(path, state)

            payoff_value = float(self.payoff(state))
            discounted_payoffs[i] = df_T * payoff_value

        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(self.n_paths))
        return price, stderr

    # American / Bermudan pricing via LSM

    def _build_american_feature_cache(self, paths: np.ndarray) -> Dict[str, Any]:
        """
        Precompute feature values along the path *prefixes* for fast American LSM.

        We special-case AsianFeature, LookbackFeature and BarrierFeature, since they are
        the building blocks of your exotics. Any other custom Feature falls back to a
        slower generic recomputation.

        Returns a cache dict used by _build_prefix_state().
        """
        n_paths, n_times = paths.shape
        t_grid = self.timegrid

        cache: Dict[str, Any] = {
            "asian": {},
            "lookback": {},
            "barrier": {},
            "fallback": set(),
        }

        if not self.features:
            return cache

        # --- AsianFeature: running average S̄_t ---
        needs_asian = any(isinstance(f, AsianFeature) for f in self.features)
        if needs_asian:
            cumsum = np.cumsum(paths, axis=1)
            denom = np.arange(1, n_times + 1, dtype=float)
            avg_paths = cumsum / denom  # broadcasting
            for feat in self.features:
                if isinstance(feat, AsianFeature):
                    cache["asian"][feat] = {"average": avg_paths}

        # --- LookbackFeature: running min/max and first time of min/max ---
        needs_lb_max = any(
            isinstance(f, LookbackFeature) and f.mode in ("max", "both") for f in self.features
        )
        needs_lb_min = any(
            isinstance(f, LookbackFeature) and f.mode in ("min", "both") for f in self.features
        )

        max_paths = max_idx_paths = None
        min_paths = min_idx_paths = None

        if needs_lb_max:
            max_paths = np.empty_like(paths)
            max_idx_paths = np.empty_like(paths, dtype=int)
            max_paths[:, 0] = paths[:, 0]
            max_idx_paths[:, 0] = 0
            for t in range(1, n_times):
                prev_max = max_paths[:, t - 1]
                cur = paths[:, t]
                improved = cur > prev_max
                max_paths[:, t] = np.where(improved, cur, prev_max)
                max_idx_paths[:, t] = np.where(improved, t, max_idx_paths[:, t - 1])

        if needs_lb_min:
            min_paths = np.empty_like(paths)
            min_idx_paths = np.empty_like(paths, dtype=int)
            min_paths[:, 0] = paths[:, 0]
            min_idx_paths[:, 0] = 0
            for t in range(1, n_times):
                prev_min = min_paths[:, t - 1]
                cur = paths[:, t]
                improved = cur < prev_min
                min_paths[:, t] = np.where(improved, cur, prev_min)
                min_idx_paths[:, t] = np.where(improved, t, min_idx_paths[:, t - 1])

        for feat in self.features:
            if isinstance(feat, LookbackFeature):
                data: Dict[str, Any] = {}
                if feat.mode in ("max", "both") and max_paths is not None:
                    data["max"] = max_paths
                    if feat.track_time:
                        data["max_idx"] = max_idx_paths
                if feat.mode in ("min", "both") and min_paths is not None:
                    data["min"] = min_paths
                    if feat.track_time:
                        data["min_idx"] = min_idx_paths
                cache["lookback"][feat] = data

        # --- BarrierFeature: barrier flags and first-hit time ---
        for feat in self.features:
            if isinstance(feat, BarrierFeature):
                level = float(feat.level)
                if feat.direction == "up":
                    if feat.strict:
                        cond = paths > level
                    else:
                        cond = paths >= level
                else:  # "down"
                    if feat.strict:
                        cond = paths < level
                    else:
                        cond = paths <= level

                # Flag: barrier hit at or before time t
                flag_paths = np.logical_or.accumulate(cond, axis=1)

                # First hit index per path
                hit_idx = np.where(cond, np.arange(n_times)[None, :], n_times)
                first_hit_idx = hit_idx.min(axis=1)
                first_hit_idx[first_hit_idx == n_times] = -1

                cache["barrier"][feat] = {
                    "flag": flag_paths,
                    "first_hit_idx": first_hit_idx,
                }

        # Any other feature → fallback (slower)
        for feat in self.features:
            if not isinstance(feat, (AsianFeature, LookbackFeature, BarrierFeature)):
                cache["fallback"].add(feat)

        return cache

    def _build_prefix_state(
        self,
        path_idx: int,
        t_idx: int,
        paths: np.ndarray,
        cache: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the state dict at time index t_idx using only the path prefix up to t_idx,
        using precomputed caches when possible.
        """
        t_grid = self.timegrid
        state: Dict[str, Any] = {}

        # 'terminal' is always interpreted as "current underlying" for American exercise
        state["terminal"] = float(paths[path_idx, t_idx])

        # AsianFeature
        asian_cache = cache["asian"]
        for feat in asian_cache.keys():
            avg_paths = asian_cache[feat]["average"]
            state["average"] = float(avg_paths[path_idx, t_idx])

        # LookbackFeature
        lookback_cache = cache["lookback"]
        for feat, data in lookback_cache.items():
            if feat.mode in ("max", "both") and "max" in data:
                state["max"] = float(data["max"][path_idx, t_idx])
                if feat.track_time and "max_idx" in data:
                    idx = int(data["max_idx"][path_idx, t_idx])
                    state["max_time"] = float(t_grid[idx])
            if feat.mode in ("min", "both") and "min" in data:
                state["min"] = float(data["min"][path_idx, t_idx])
                if feat.track_time and "min_idx" in data:
                    idx = int(data["min_idx"][path_idx, t_idx])
                    state["min_time"] = float(t_grid[idx])

        # BarrierFeature
        barrier_cache = cache["barrier"]
        for feat, data in barrier_cache.items():
            flag_paths = data["flag"]
            first_hit_idx = data["first_hit_idx"]

            key_flag = f"{feat.name}_flag"
            flag = bool(flag_paths[path_idx, t_idx])
            state[key_flag] = flag

            if feat.track_time:
                fh_idx = int(first_hit_idx[path_idx])
                if fh_idx == -1 or fh_idx > t_idx:
                    state[f"{feat.name}_time"] = math.inf
                else:
                    state[f"{feat.name}_time"] = float(t_grid[fh_idx])

        # Fallback features: recompute from prefix path (slower but generic)
        for feat in cache["fallback"]:
            prefix_path = paths[path_idx, : t_idx + 1]
            sub_state: Dict[str, Any] = {"t_grid": t_grid[: t_idx + 1]}
            feat.apply(prefix_path, sub_state)
            for k, v in sub_state.items():
                if k not in state:
                    state[k] = v

        return state

    def _build_regression_basis(
        self,
        states: Sequence[Dict[str, Any]],
        basis_keys: Sequence[str],
    ) -> np.ndarray:
        """
        Build regression design matrix for LSM.

        Basis: [1, x_j, x_j^2] for each required key x_j in basis_keys.
        """
        n = len(states)
        d = len(basis_keys)
        # 1 + d + d columns
        X = np.empty((n, 1 + d + d), dtype=float)

        for i, st in enumerate(states):
            xs = [float(st[k]) for k in basis_keys]
            X[i, 0] = 1.0
            X[i, 1 : 1 + d] = xs
            X[i, 1 + d :] = [x * x for x in xs]

        return X

    def _price_american_lsm(self) -> Tuple[float, float]:
        """
        Longstaff–Schwartz regression for American / Bermudan options,
        with full support for path-dependent payoffs via Features.

        Markov state at exercise time t_i is the vector of keys in payoff.requires,
        computed from the path prefix up to t_i.
        """
        paths = self._simulate_paths(randnumber_key="american")
        n_paths, n_times = paths.shape
        t_grid = self.timegrid
        discount = self._discount_grid

        # 1) European-style terminal payoff as fallback continuation value
        Y = np.empty(n_paths, dtype=float)
        df_T = discount[-1]

        for i in range(n_paths):
            path = paths[i, :]
            state: Dict[str, Any] = {
                "t_grid": t_grid,
                "terminal": float(path[-1]),
            }
            for feat in self.features:
                feat.apply(path, state)
            Y[i] = df_T * float(self.payoff(state))

        alive = np.ones(n_paths, dtype=bool)

        # 2) Exercise indices (American, Bermudan, etc.)
        exercise_indices = [
            idx for idx in range(n_times) if self.exercise.exercise_time(idx, t_grid)
        ]

        if not exercise_indices:
            # No early exercise → European
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        # Remove maturity if exercise policy also flags it; we already account for exercise at T via Y
        exercise_indices = sorted(set(exercise_indices))
        exercise_indices = [i for i in exercise_indices if i < n_times - 1]

        if not exercise_indices:
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        # 3) Precompute feature states for prefixes
        cache = self._build_american_feature_cache(paths)

        # Basis keys: what payoff actually depends on
        basis_keys = sorted(self.payoff.requires)

        # 4) Backward induction over exercise dates
        for t_idx in reversed(exercise_indices):
            df_t = discount[t_idx]

            alive_idx = np.where(alive)[0]
            if alive_idx.size == 0:
                break

            # Build states and immediate payoff for alive paths
            immediate = np.zeros(n_paths, dtype=float)
            states_alive: Dict[int, Dict[str, Any]] = {}

            for i_pos, i_path in enumerate(alive_idx):
                st = self._build_prefix_state(i_path, t_idx, paths, cache)
                states_alive[i_path] = st
                immediate[i_path] = df_t * float(self.payoff(st))

            # In-the-money alive paths
            itm_mask_global = (immediate > 0.0) & alive
            idx_itm = np.where(itm_mask_global)[0]
            if idx_itm.size == 0:
                continue

            # Regression design matrix on ITM & alive paths
            states_itm = [states_alive[i_path] for i_path in idx_itm]
            X = self._build_regression_basis(states_itm, basis_keys)
            y = Y[idx_itm]  # continuation value already discounted to t=0

            # If too few points, skip regression at this date
            if X.shape[0] <= X.shape[1]:
                continue

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            continuation_est = X @ beta  # estimated E[Y | state_t]

            # Exercise decision: immediate vs continuation
            exercise_now = np.zeros(n_paths, dtype=bool)
            for j, path_idx in enumerate(idx_itm):
                exercise_now[path_idx] = immediate[path_idx] > continuation_est[j]

            # Update Y and alive flags
            Y[exercise_now] = immediate[exercise_now]
            alive[exercise_now] = False

        price = float(np.mean(Y))
        stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
        return price, stderr


# #######################
# ### ENGINE GPT CODE ###
# #######################

#######################################################################################################################


############################
### OPTION CONFIGURATION ###
############################

# time grid
t_grid = np.linspace(0.0, 1.0, 365)

# ----------------------------------------------------------------------
# American knock-in barrier Asian call configuration
# ----------------------------------------------------------------------
K = 100.0
barrier_level = 120.0  # up-and-in level

# Base Asian call on the arithmetic average
asian_call = Vanilla(key="average", k=K, side="call")

# Path features: arithmetic average + knock-in barrier
features = [
    AsianFeature(),
    BarrierFeature(
        name="KI",
        level=barrier_level,
        direction="up",  # up-and-in
        strict=False,  # >= barrier_level
        track_time=True,
    ),
]

# Knock-in wrapper: pays only if KI_flag was triggered at some point
payoff = KnockInAllN(inner=asian_call, keys=["KI_flag"])

# American exercise (early exercise allowed on the whole grid + enforced at maturity)
exercise = American()

# === QuantLib / Heston set-up ===
calendar = ql.TARGET()
day_count = ql.Actual365Fixed()
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

S0 = 100.0
r = 0.01
q = 0.00
v0 = 0.04
kappa = 1.5
theta = 0.07
sigma = 0.4
rho = -0.7

s0_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))

# ---- NEW: Numba-based Heston QE sampler wrapped as a Model ----

numba_sampler = NumbaHestonSampler(
    S0=S0,
    v0=v0,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    rho=rho,
    r=r,
    q=q,
    v_floor=1e-12,
)

# Reuse your BatesModel as a generic NumPy model wrapper
model = BatesModel(
    sampler=numba_sampler,
    S0=S0,
    r=r,
    q=q,
)


# --- Monte Carlo config ---
cfg = MonteCarloConfig(
    n_paths=1000, timegrid=t_grid, seed=24976537, use_crn=False, antithetic=False
)

engine = MonteCarloEngine(
    model=model,
    payoff=payoff,
    features=features,
    exercise=exercise,
    config=cfg,
)

start = time.perf_counter()

price, stderr = engine.price()
paths = engine._simulate_paths()
end = time.perf_counter()

print("MC price:", price, "std err:", stderr)
print(f"Total time pricing: {end - start:.4f} seconds")


########################################################################################################################


paths = engine._simulate_paths()  # shape (n_paths, len(t_grid))

plt.figure(figsize=(8, 5))

n_plot = min(1000, paths.shape[0])

for i in range(n_plot):
    plt.plot(t_grid, paths[i, :], alpha=0.15, linewidth=2)

plt.xlabel("Time (years)")
plt.ylabel("Underlying price")
plt.title(f"Sample Heston paths (showing {n_plot} of {paths.shape[0]} simulated)")
plt.grid(True)
plt.tight_layout()
plt.show()
