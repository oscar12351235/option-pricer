from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import MonteCarloConfig
from .exercise import American, European, Exercise
from .features.core import AsianFeature, BarrierFeature, Feature, LookbackFeature
from .models.base import Model
from .payoffs.core import Payoff


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
      including path-dependent products through Features.
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
        if self.timegrid.size < 2:
            raise ValueError("timegrid must contain at least two time points")

        self.n_paths = int(self.cfg.n_paths)
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if self.cfg.antithetic and self.n_paths % 2 != 0:
            raise ValueError("n_paths must be even when using antithetic variates")

        self.rng = np.random.default_rng(self.cfg.seed)
        self._discount_grid = np.array(
            [self.model.discount(float(t)) for t in self.timegrid], dtype=float
        )

        self._z_crn = None
        supports_crn = getattr(self.model, "supports_crn", False)
        supports_antithetic = getattr(self.model, "supports_antithetic", False)
        if self.cfg.use_crn and supports_crn:
            n_steps = self.timegrid.size - 1
            if self.cfg.antithetic and supports_antithetic:
                n_half = self.n_paths // 2
                z_half = self.rng.standard_normal(size=(n_steps, n_half, 2))
                self._z_crn = np.concatenate([z_half, -z_half], axis=1)
            else:
                self._z_crn = self.rng.standard_normal(size=(n_steps, self.n_paths, 2))

    def price(self) -> tuple[float, float]:
        if isinstance(self.exercise, European):
            return self._price_european()
        if isinstance(self.exercise, American):
            return self._price_american_lsm()
        return self._price_american_lsm()

    def _simulate_paths(self, randnumber_key: str = "base") -> np.ndarray:
        sampler = self.model.path_sampler()
        supports_crn = getattr(self.model, "supports_crn", False)
        supports_antithetic = getattr(self.model, "supports_antithetic", False)

        z = None
        if supports_crn:
            if self._z_crn is not None:
                z = self._z_crn
            elif self.cfg.antithetic and supports_antithetic:
                n_steps = self.timegrid.size - 1
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

        expected_shape = (self.n_paths, self.timegrid.size)
        if paths.shape != expected_shape:
            raise ValueError(
                f"Model sampler returned array of shape {paths.shape}, expected {expected_shape}"
            )
        return paths

    def _price_european(self) -> tuple[float, float]:
        paths = self._simulate_paths(randnumber_key="european")
        df_t = self._discount_grid[-1]

        discounted_payoffs = np.empty(self.n_paths, dtype=float)
        for i in range(self.n_paths):
            path = paths[i, :]
            state: dict[str, Any] = {"t_grid": self.timegrid, "terminal": float(path[-1])}
            for feat in self.features:
                feat.apply(path, state)
            discounted_payoffs[i] = df_t * float(self.payoff(state))

        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(self.n_paths))
        return price, stderr

    def _build_american_feature_cache(self, paths: np.ndarray) -> dict[str, Any]:
        n_paths, n_times = paths.shape

        cache: dict[str, Any] = {"asian": {}, "lookback": {}, "barrier": {}, "fallback": set()}
        if not self.features:
            return cache

        needs_asian = any(isinstance(f, AsianFeature) for f in self.features)
        if needs_asian:
            cumsum = np.cumsum(paths, axis=1)
            denom = np.arange(1, n_times + 1, dtype=float)
            avg_paths = cumsum / denom
            for feat in self.features:
                if isinstance(feat, AsianFeature):
                    cache["asian"][feat] = {"average": avg_paths}

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
                data: dict[str, Any] = {}
                if feat.mode in ("max", "both") and max_paths is not None:
                    data["max"] = max_paths
                    if feat.track_time:
                        data["max_idx"] = max_idx_paths
                if feat.mode in ("min", "both") and min_paths is not None:
                    data["min"] = min_paths
                    if feat.track_time:
                        data["min_idx"] = min_idx_paths
                cache["lookback"][feat] = data

        for feat in self.features:
            if isinstance(feat, BarrierFeature):
                level = float(feat.level)
                if feat.direction == "up":
                    cond = paths > level if feat.strict else paths >= level
                else:
                    cond = paths < level if feat.strict else paths <= level

                flag_paths = np.logical_or.accumulate(cond, axis=1)

                hit_idx = np.where(cond, np.arange(n_times)[None, :], n_times)
                first_hit_idx = hit_idx.min(axis=1)
                first_hit_idx[first_hit_idx == n_times] = -1

                cache["barrier"][feat] = {"flag": flag_paths, "first_hit_idx": first_hit_idx}

        for feat in self.features:
            if not isinstance(feat, (AsianFeature, LookbackFeature, BarrierFeature)):
                cache["fallback"].add(feat)

        return cache

    def _build_prefix_state(
        self, path_idx: int, t_idx: int, paths: np.ndarray, cache: dict[str, Any]
    ) -> dict[str, Any]:
        t_grid = self.timegrid
        state: dict[str, Any] = {"terminal": float(paths[path_idx, t_idx])}

        for feat in cache["asian"].keys():
            avg_paths = cache["asian"][feat]["average"]
            state["average"] = float(avg_paths[path_idx, t_idx])

        for feat, data in cache["lookback"].items():
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

        for feat, data in cache["barrier"].items():
            key_flag = f"{feat.name}_flag"
            state[key_flag] = bool(data["flag"][path_idx, t_idx])

            if feat.track_time:
                first_hit_idx = int(data["first_hit_idx"][path_idx])
                if first_hit_idx == -1 or first_hit_idx > t_idx:
                    state[f"{feat.name}_time"] = math.inf
                else:
                    state[f"{feat.name}_time"] = float(t_grid[first_hit_idx])

        for feat in cache["fallback"]:
            prefix_path = paths[path_idx, : t_idx + 1]
            sub_state: dict[str, Any] = {"t_grid": t_grid[: t_idx + 1]}
            feat.apply(prefix_path, sub_state)
            for key, value in sub_state.items():
                if key not in state:
                    state[key] = value

        return state

    @staticmethod
    def _build_regression_basis(
        states: Sequence[dict[str, Any]], basis_keys: Sequence[str]
    ) -> np.ndarray:
        n = len(states)
        d = len(basis_keys)
        X = np.empty((n, 1 + d + d), dtype=float)
        X[:, 0] = 1.0

        for col, key in enumerate(basis_keys):
            x = np.fromiter((float(st[key]) for st in states), dtype=float, count=n)
            X[:, 1 + col] = x
            X[:, 1 + d + col] = x * x

        return X

    def _price_american_lsm(self) -> tuple[float, float]:
        paths = self._simulate_paths(randnumber_key="american")
        n_paths, n_times = paths.shape
        t_grid = self.timegrid
        discount = self._discount_grid

        y = np.empty(n_paths, dtype=float)
        df_t = discount[-1]
        for i in range(n_paths):
            path = paths[i, :]
            state: dict[str, Any] = {"t_grid": t_grid, "terminal": float(path[-1])}
            for feat in self.features:
                feat.apply(path, state)
            y[i] = df_t * float(self.payoff(state))

        alive = np.ones(n_paths, dtype=bool)

        exercise_indices = [
            idx for idx in range(n_times) if self.exercise.exercise_time(idx, t_grid)
        ]
        if not exercise_indices:
            price = float(np.mean(y))
            stderr = float(np.std(y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        exercise_indices = sorted(set(exercise_indices))
        exercise_indices = [idx for idx in exercise_indices if idx < n_times - 1]
        if not exercise_indices:
            price = float(np.mean(y))
            stderr = float(np.std(y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        cache = self._build_american_feature_cache(paths)
        basis_keys = sorted(self.payoff.requires)

        for t_idx in reversed(exercise_indices):
            df_t = discount[t_idx]
            alive_idx = np.where(alive)[0]
            if alive_idx.size == 0:
                break

            immediate = np.zeros(n_paths, dtype=float)
            states_alive: dict[int, dict[str, Any]] = {}
            for i_path in alive_idx:
                state = self._build_prefix_state(int(i_path), t_idx, paths, cache)
                states_alive[int(i_path)] = state
                immediate[int(i_path)] = df_t * float(self.payoff(state))

            itm_mask_global = (immediate > 0.0) & alive
            idx_itm = np.where(itm_mask_global)[0]
            if idx_itm.size == 0:
                continue

            states_itm = [states_alive[int(i_path)] for i_path in idx_itm]
            X = self._build_regression_basis(states_itm, basis_keys)

            if X.shape[0] <= X.shape[1]:
                continue

            beta, *_ = np.linalg.lstsq(X, y[idx_itm], rcond=None)
            continuation_est = X @ beta
            exercise_now = immediate[idx_itm] > continuation_est
            exercise_paths = idx_itm[exercise_now]

            y[exercise_paths] = immediate[exercise_paths]
            alive[exercise_paths] = False

        price = float(np.mean(y))
        stderr = float(np.std(y, ddof=1) / math.sqrt(n_paths))
        return price, stderr
