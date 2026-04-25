from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import MonteCarloConfig
from .exercise import American, European
from .features.core import AsianFeature, BarrierFeature, Feature, LookbackFeature
from .models.base import Model
from .payoffs.core import (
    AsianFloatingVanilla,
    Digital,
    KnockInAllN,
    KnockInKnockOut,
    KnockOutAnyN,
    Payoff,
    Vanilla,
)


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
    def __init__(
        self,
        model: Model,
        payoff: Payoff,
        features: Sequence[Feature],
        exercise,
        config: MonteCarloConfig,
    ):
        self.model = model
        self.payoff = payoff
        self.features = list(features)
        self.exercise = exercise
        self.cfg = config
        self._required_keys = set(self.payoff.requires)

        payoff_feature_key(payoff, self.features)
        self._active_features = [
            feat for feat in self.features if self._required_keys.intersection(feat.provides)
        ]

        self.timegrid = np.asarray(self.cfg.timegrid, dtype=float)
        if self.timegrid.ndim != 1:
            raise ValueError("timegrid must be 1D")
        if len(self.timegrid) < 2:
            raise ValueError("timegrid must contain at least two time points")

        self.n_paths = int(self.cfg.n_paths)
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.cfg.antithetic and self.n_paths % 2 != 0:
            raise ValueError("n_paths must be even when using antithetic variates")

        self.rng = np.random.default_rng(self.cfg.seed)
        self._discount_grid = np.array(
            [self.model.discount(float(t)) for t in self.timegrid],
            dtype=float,
        )

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

        expected_shape = (self.n_paths, len(self.timegrid))
        if paths.shape != expected_shape:
            raise ValueError(
                f"Model sampler returned array of shape {paths.shape}, "
                f"expected {expected_shape}"
            )
        return paths

    def _price_european(self) -> tuple[float, float]:
        paths = self._simulate_paths(randnumber_key="european")
        df_t = self._discount_grid[-1]
        key_arrays = self._build_vectorized_key_arrays(paths)

        if key_arrays is not None and self._is_vectorized_payoff(self.payoff):
            discounted = df_t * self._evaluate_payoff_vectorized(
                self.payoff,
                key_arrays,
                t_idx=len(self.timegrid) - 1,
            )
        else:
            discounted = np.empty(self.n_paths, dtype=float)
            for i in range(self.n_paths):
                path = paths[i, :]
                state: dict[str, Any] = {
                    "t_grid": self.timegrid,
                    "terminal": float(path[-1]),
                }
                for feat in self._active_features:
                    feat.apply(path, state)
                discounted[i] = df_t * float(self.payoff(state))

        price = float(np.mean(discounted))
        stderr = float(np.std(discounted, ddof=1) / math.sqrt(self.n_paths))
        return price, stderr

    def _build_american_feature_cache(self, paths: np.ndarray) -> dict[str, Any]:
        n_paths, n_times = paths.shape

        cache: dict[str, Any] = {
            "asian": {},
            "lookback": {},
            "barrier": {},
            "fallback": set(),
        }

        if not self._active_features:
            return cache

        needs_asian = any(isinstance(f, AsianFeature) for f in self._active_features)
        if needs_asian:
            cumsum = np.cumsum(paths, axis=1)
            denom = np.arange(1, n_times + 1, dtype=float)
            avg_paths = cumsum / denom
            for feat in self._active_features:
                if isinstance(feat, AsianFeature):
                    cache["asian"][feat] = {"average": avg_paths}

        needs_lb_max = any(
            isinstance(f, LookbackFeature) and f.mode in ("max", "both")
            for f in self._active_features
        )
        needs_lb_min = any(
            isinstance(f, LookbackFeature) and f.mode in ("min", "both")
            for f in self._active_features
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

        for feat in self._active_features:
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

        for feat in self._active_features:
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
                cache["barrier"][feat] = {
                    "flag": flag_paths,
                    "first_hit_idx": first_hit_idx,
                }

        for feat in self._active_features:
            if not isinstance(feat, AsianFeature | LookbackFeature | BarrierFeature):
                cache["fallback"].add(feat)

        return cache

    def _build_prefix_state(
        self,
        path_idx: int,
        t_idx: int,
        paths: np.ndarray,
        cache: dict[str, Any],
    ) -> dict[str, Any]:
        t_grid = self.timegrid
        state: dict[str, Any] = {"terminal": float(paths[path_idx, t_idx])}

        for feat in cache["asian"]:
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
            flag = bool(data["flag"][path_idx, t_idx])
            state[f"{feat.name}_flag"] = flag
            if feat.track_time:
                hit_idx = int(data["first_hit_idx"][path_idx])
                if hit_idx == -1 or hit_idx > t_idx:
                    state[f"{feat.name}_time"] = math.inf
                else:
                    state[f"{feat.name}_time"] = float(t_grid[hit_idx])

        for feat in cache["fallback"]:
            prefix_path = paths[path_idx, : t_idx + 1]
            sub_state: dict[str, Any] = {"t_grid": t_grid[: t_idx + 1]}
            feat.apply(prefix_path, sub_state)
            for key, value in sub_state.items():
                if key not in state:
                    state[key] = value

        return state

    def _build_regression_basis(
        self,
        states: Sequence[dict[str, Any]],
        basis_keys: Sequence[str],
    ) -> np.ndarray:
        n = len(states)
        d = len(basis_keys)
        X = np.empty((n, 1 + d + d), dtype=float)
        for i, state in enumerate(states):
            xs = [float(state[key]) for key in basis_keys]
            X[i, 0] = 1.0
            X[i, 1 : 1 + d] = xs
            X[i, 1 + d :] = [x * x for x in xs]
        return X

    def _build_regression_basis_from_arrays(
        self,
        key_arrays: dict[str, np.ndarray],
        basis_keys: Sequence[str],
        row_idx: np.ndarray,
        t_idx: int,
    ) -> np.ndarray:
        d = len(basis_keys)
        X = np.empty((len(row_idx), 1 + d + d), dtype=float)
        X[:, 0] = 1.0

        for j, key in enumerate(basis_keys):
            values = np.asarray(key_arrays[key])[row_idx, t_idx].astype(float)
            X[:, 1 + j] = values
            X[:, 1 + d + j] = values * values

        return X

    def _price_american_lsm(self) -> tuple[float, float]:
        paths = self._simulate_paths(randnumber_key="american")
        n_paths, n_times = paths.shape

        key_arrays = self._build_vectorized_key_arrays(paths)
        if key_arrays is not None and self._is_vectorized_payoff(self.payoff):
            return self._price_american_lsm_vectorized(paths, key_arrays)

        Y = np.empty(n_paths, dtype=float)
        df_T = self._discount_grid[-1]
        for i in range(n_paths):
            path = paths[i, :]
            state: dict[str, Any] = {"t_grid": self.timegrid, "terminal": float(path[-1])}
            for feat in self._active_features:
                feat.apply(path, state)
            Y[i] = df_T * float(self.payoff(state))

        alive = np.ones(n_paths, dtype=bool)
        exercise_indices = [
            idx for idx in range(n_times) if self.exercise.exercise_time(idx, self.timegrid)
        ]
        if not exercise_indices:
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        exercise_indices = sorted(set(exercise_indices))
        exercise_indices = [idx for idx in exercise_indices if idx < n_times - 1]
        if not exercise_indices:
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        cache = self._build_american_feature_cache(paths)
        basis_keys = sorted(self.payoff.requires)

        for t_idx in reversed(exercise_indices):
            df_t = self._discount_grid[t_idx]
            alive_idx = np.where(alive)[0]
            if alive_idx.size == 0:
                break

            immediate = np.zeros(n_paths, dtype=float)
            states_alive: dict[int, dict[str, Any]] = {}

            for path_idx in alive_idx:
                state = self._build_prefix_state(path_idx, t_idx, paths, cache)
                states_alive[path_idx] = state
                immediate[path_idx] = df_t * float(self.payoff(state))

            idx_itm = np.where((immediate > 0.0) & alive)[0]
            if idx_itm.size == 0:
                continue

            states_itm = [states_alive[path_idx] for path_idx in idx_itm]
            X = self._build_regression_basis(states_itm, basis_keys)
            y = Y[idx_itm]

            if X.shape[0] <= X.shape[1]:
                continue

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            continuation_est = X @ beta

            exercise_now = np.zeros(n_paths, dtype=bool)
            for j, path_idx in enumerate(idx_itm):
                exercise_now[path_idx] = immediate[path_idx] > continuation_est[j]

            Y[exercise_now] = immediate[exercise_now]
            alive[exercise_now] = False

        price = float(np.mean(Y))
        stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
        return price, stderr

    def _price_american_lsm_vectorized(
        self,
        paths: np.ndarray,
        key_arrays: dict[str, np.ndarray],
    ) -> tuple[float, float]:
        n_paths, n_times = paths.shape
        Y = self._discount_grid[-1] * self._evaluate_payoff_vectorized(
            self.payoff,
            key_arrays,
            t_idx=n_times - 1,
        )

        alive = np.ones(n_paths, dtype=bool)
        exercise_indices = [
            idx for idx in range(n_times) if self.exercise.exercise_time(idx, self.timegrid)
        ]
        if not exercise_indices:
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        exercise_indices = sorted(set(exercise_indices))
        exercise_indices = [idx for idx in exercise_indices if idx < n_times - 1]
        if not exercise_indices:
            price = float(np.mean(Y))
            stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
            return price, stderr

        basis_keys = sorted(self.payoff.requires)

        for t_idx in reversed(exercise_indices):
            alive_idx = np.where(alive)[0]
            if alive_idx.size == 0:
                break

            immediate = self._discount_grid[t_idx] * self._evaluate_payoff_vectorized(
                self.payoff,
                key_arrays,
                t_idx=t_idx,
            )
            idx_itm = alive_idx[immediate[alive_idx] > 0.0]
            if idx_itm.size == 0:
                continue

            X = self._build_regression_basis_from_arrays(key_arrays, basis_keys, idx_itm, t_idx)
            y = Y[idx_itm]
            if X.shape[0] <= X.shape[1]:
                continue

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            continuation_est = X @ beta
            exercise_now = idx_itm[immediate[idx_itm] > continuation_est]
            if exercise_now.size == 0:
                continue

            Y[exercise_now] = immediate[exercise_now]
            alive[exercise_now] = False

        price = float(np.mean(Y))
        stderr = float(np.std(Y, ddof=1) / math.sqrt(n_paths))
        return price, stderr

    def _build_vectorized_key_arrays(self, paths: np.ndarray) -> dict[str, np.ndarray] | None:
        key_arrays: dict[str, np.ndarray] = {"terminal": np.asarray(paths, dtype=float)}
        n_paths, n_times = paths.shape

        for feat in self._active_features:
            if isinstance(feat, AsianFeature):
                cumsum = np.cumsum(paths, axis=1)
                denom = np.arange(1, n_times + 1, dtype=float)
                key_arrays["average"] = cumsum / denom
                continue

            if isinstance(feat, LookbackFeature):
                if feat.mode in ("max", "both"):
                    max_vals = np.maximum.accumulate(paths, axis=1)
                    key_arrays["max"] = max_vals
                    if feat.track_time:
                        max_idx = np.empty((n_paths, n_times), dtype=int)
                        max_idx[:, 0] = 0
                        for t in range(1, n_times):
                            improved = paths[:, t] > max_vals[:, t - 1]
                            max_idx[:, t] = np.where(improved, t, max_idx[:, t - 1])
                        key_arrays["max_time"] = self.timegrid[max_idx]
                if feat.mode in ("min", "both"):
                    min_vals = np.minimum.accumulate(paths, axis=1)
                    key_arrays["min"] = min_vals
                    if feat.track_time:
                        min_idx = np.empty((n_paths, n_times), dtype=int)
                        min_idx[:, 0] = 0
                        for t in range(1, n_times):
                            improved = paths[:, t] < min_vals[:, t - 1]
                            min_idx[:, t] = np.where(improved, t, min_idx[:, t - 1])
                        key_arrays["min_time"] = self.timegrid[min_idx]
                continue

            if isinstance(feat, BarrierFeature):
                level = float(feat.level)
                if feat.direction == "up":
                    cond = paths > level if feat.strict else paths >= level
                else:
                    cond = paths < level if feat.strict else paths <= level

                flags = np.logical_or.accumulate(cond, axis=1)
                key_arrays[f"{feat.name}_flag"] = flags

                if feat.track_time:
                    first_hit = np.full((n_paths, n_times), np.inf, dtype=float)
                    hit_once = np.zeros(n_paths, dtype=bool)
                    hit_time = np.full(n_paths, np.inf, dtype=float)
                    for t in range(n_times):
                        new_hit = cond[:, t] & ~hit_once
                        hit_once[new_hit] = True
                        hit_time[new_hit] = float(self.timegrid[t])
                        first_hit[:, t] = np.where(hit_once, hit_time, np.inf)
                    key_arrays[f"{feat.name}_time"] = first_hit
                continue

            return None

        return key_arrays

    def _is_vectorized_payoff(self, payoff: Payoff) -> bool:
        if isinstance(payoff, Vanilla | Digital | AsianFloatingVanilla):
            return True
        if isinstance(payoff, KnockOutAnyN | KnockInAllN | KnockInKnockOut):
            return self._is_vectorized_payoff(payoff.inner)
        return False

    def _evaluate_payoff_vectorized(
        self,
        payoff: Payoff,
        key_arrays: dict[str, np.ndarray],
        t_idx: int,
    ) -> np.ndarray:
        if isinstance(payoff, Vanilla):
            x = np.asarray(key_arrays[payoff.key])[:, t_idx].astype(float)
            if payoff.side == "call":
                return np.maximum(x - payoff.k, 0.0)
            return np.maximum(payoff.k - x, 0.0)

        if isinstance(payoff, Digital):
            x = np.asarray(key_arrays[payoff.key])[:, t_idx].astype(float)
            if payoff.side == "call":
                hit = x > payoff.k
            else:
                hit = x < payoff.k
            if payoff.variant == "cash":
                return np.where(hit, payoff.cash, 0.0)
            return np.where(hit, x, 0.0)

        if isinstance(payoff, AsianFloatingVanilla):
            x = np.asarray(key_arrays[payoff.key1])[:, t_idx].astype(float)
            k = np.asarray(key_arrays[payoff.key2])[:, t_idx].astype(float)
            if payoff.side == "call":
                return np.maximum(x - k, 0.0)
            return np.maximum(k - x, 0.0)

        if isinstance(payoff, KnockOutAnyN):
            inner = self._evaluate_payoff_vectorized(payoff.inner, key_arrays, t_idx)
            knocked_out = np.zeros_like(inner, dtype=bool)
            for key in payoff.keys:
                knocked_out |= np.asarray(key_arrays[key])[:, t_idx].astype(bool)
            return np.where(knocked_out, 0.0, inner)

        if isinstance(payoff, KnockInAllN):
            inner = self._evaluate_payoff_vectorized(payoff.inner, key_arrays, t_idx)
            knocked_in = np.ones_like(inner, dtype=bool)
            for key in payoff.keys:
                knocked_in &= np.asarray(key_arrays[key])[:, t_idx].astype(bool)
            return np.where(knocked_in, inner, 0.0)

        if isinstance(payoff, KnockInKnockOut):
            inner = self._evaluate_payoff_vectorized(payoff.inner, key_arrays, t_idx)
            knocked_in = np.asarray(key_arrays[payoff.ki_key])[:, t_idx].astype(bool)
            knocked_out = np.asarray(key_arrays[payoff.ko_key])[:, t_idx].astype(bool)
            return np.where(knocked_in & ~knocked_out, inner, 0.0)

        raise TypeError(f"Unsupported vectorized payoff type: {type(payoff).__name__}")
