from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

config_mod = importlib.import_module("option-pricer.config")
engine_mod = importlib.import_module("option-pricer.engine")
exercise_mod = importlib.import_module("option-pricer.exercise")
features_mod = importlib.import_module("option-pricer.features.core")
models_mod = importlib.import_module("option-pricer.models.bates")
payoffs_mod = importlib.import_module("option-pricer.payoffs.core")

MonteCarloConfig = config_mod.MonteCarloConfig
MonteCarloEngine = engine_mod.MonteCarloEngine
American = exercise_mod.American
European = exercise_mod.European
Feature = features_mod.Feature
AsianFeature = features_mod.AsianFeature
BarrierFeature = features_mod.BarrierFeature
LookbackFeature = features_mod.LookbackFeature
BatesModel = models_mod.BatesModel
KnockInAllN = payoffs_mod.KnockInAllN
Payoff = payoffs_mod.Payoff
Vanilla = payoffs_mod.Vanilla
Digital = payoffs_mod.Digital
AsianFloatingVanilla = payoffs_mod.AsianFloatingVanilla
KnockOutAnyN = payoffs_mod.KnockOutAnyN
KnockInKnockOut = payoffs_mod.KnockInKnockOut


class ConstantSampler:
    supports_crn = False
    supports_antithetic = False

    def __init__(self, s0: float):
        self.s0 = float(s0)

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
        return np.full((int(n_paths), len(timegrid)), self.s0, dtype=float)


class VectorizedGBMSampler:
    supports_crn = True
    supports_antithetic = True

    def __init__(self, s0: float, mu: float, sigma: float):
        self.s0 = float(s0)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
        n_paths = int(n_paths)
        timegrid = np.asarray(timegrid, dtype=float)
        n_steps = len(timegrid) - 1

        if z is None:
            z = randnumbergenerator.standard_normal(size=(n_steps, n_paths, 2))

        dt = np.diff(timegrid)[:, None]
        shocks = z[:, :, 0]
        drift = (self.mu - 0.5 * self.sigma * self.sigma) * dt
        diffusion = self.sigma * np.sqrt(dt) * shocks
        log_increments = drift + diffusion
        log_paths = np.cumsum(log_increments, axis=0)

        paths = np.empty((n_paths, len(timegrid)), dtype=float)
        paths[:, 0] = self.s0
        paths[:, 1:] = self.s0 * np.exp(log_paths.T)
        return paths


class BadShapeSampler:
    supports_crn = False
    supports_antithetic = False

    def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
        return np.zeros((int(n_paths),), dtype=float)


class CustomSpreadFeature(Feature):
    @property
    def provides(self):
        return frozenset({"spread"})

    def apply(self, path, state: dict) -> None:
        state["spread"] = float(path[-1] - path[0])


class NeverExercise:
    def exercise_time(self, timeindex, timegrid):
        return False


def test_european_constant_path_price_is_discounted_intrinsic_value():
    model = BatesModel(sampler=ConstantSampler(s0=100.0), S0=100.0, r=0.05, q=0.0)
    payoff = Vanilla(key="terminal", k=90.0, side="call")
    cfg = MonteCarloConfig(n_paths=512, timegrid=np.linspace(0.0, 1.0, 8), seed=7)

    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=[],
        exercise=European(),
        config=cfg,
    )
    price, stderr = engine.price()

    expected = math.exp(-0.05) * 10.0
    assert price == pytest.approx(expected, rel=0.0, abs=1e-12)
    assert stderr == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_engine_rejects_missing_required_feature_keys():
    model = BatesModel(sampler=ConstantSampler(s0=100.0), S0=100.0, r=0.02, q=0.0)
    payoff = Vanilla(key="average", k=100.0, side="call")
    cfg = MonteCarloConfig(n_paths=128, timegrid=np.linspace(0.0, 1.0, 6), seed=3)

    with pytest.raises(ValueError, match="requires keys"):
        MonteCarloEngine(
            model=model,
            payoff=payoff,
            features=[],
            exercise=European(),
            config=cfg,
        )


def test_american_asian_knockin_is_reproducible_with_crn_and_antithetic():
    sampler = VectorizedGBMSampler(s0=100.0, mu=0.03, sigma=0.25)
    model = BatesModel(sampler=sampler, S0=100.0, r=0.01, q=0.0)
    payoff = KnockInAllN(inner=Vanilla(key="average", k=100.0, side="call"), keys=["KI_flag"])
    features = [
        AsianFeature(),
        BarrierFeature(name="KI", level=115.0, direction="up", strict=False, track_time=True),
    ]
    cfg = MonteCarloConfig(
        n_paths=600,
        timegrid=np.linspace(0.0, 1.0, 25),
        seed=11,
        use_crn=True,
        antithetic=True,
    )

    engine_1 = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=American(),
        config=cfg,
    )
    engine_2 = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=American(),
        config=cfg,
    )

    price_1, stderr_1 = engine_1.price()
    price_2, stderr_2 = engine_2.price()

    assert math.isfinite(price_1)
    assert math.isfinite(stderr_1)
    assert price_1 >= 0.0
    assert stderr_1 >= 0.0
    assert price_1 == pytest.approx(price_2, rel=0.0, abs=1e-12)
    assert stderr_1 == pytest.approx(stderr_2, rel=0.0, abs=1e-12)


def test_engine_validates_sampler_output_shape():
    model = BatesModel(sampler=BadShapeSampler(), S0=100.0, r=0.0, q=0.0)
    payoff = Vanilla(key="terminal", k=95.0, side="call")
    cfg = MonteCarloConfig(n_paths=10, timegrid=np.linspace(0.0, 1.0, 4), seed=1)
    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=[],
        exercise=European(),
        config=cfg,
    )

    with pytest.raises(ValueError, match="returned array of shape"):
        engine.price()


def test_vectorized_payoff_evaluator_supports_all_builtin_wrappers():
    model = BatesModel(sampler=ConstantSampler(s0=100.0), S0=100.0, r=0.0, q=0.0)
    cfg = MonteCarloConfig(n_paths=2, timegrid=np.array([0.0, 1.0]), seed=0)
    engine = MonteCarloEngine(
        model=model,
        payoff=Vanilla(key="terminal", k=100.0, side="call"),
        features=[],
        exercise=European(),
        config=cfg,
    )
    key_arrays = {
        "terminal": np.array([[90.0, 110.0], [105.0, 95.0]]),
        "average": np.array([[95.0, 100.0], [102.0, 98.0]]),
        "KI_flag": np.array([[False, True], [False, True]]),
        "KO_flag": np.array([[False, False], [True, False]]),
    }

    assert np.allclose(
        engine._evaluate_payoff_vectorized(
            Vanilla(key="terminal", k=100.0, side="call"),
            key_arrays,
            t_idx=1,
        ),
        np.array([10.0, 0.0]),
    )
    assert np.allclose(
        engine._evaluate_payoff_vectorized(
            Digital(key="terminal", k=100.0, side="put", cash=2.0, variant="asset"),
            key_arrays,
            t_idx=1,
        ),
        np.array([0.0, 95.0]),
    )
    assert np.allclose(
        engine._evaluate_payoff_vectorized(
            AsianFloatingVanilla(key1="terminal", key2="average", side="call"),
            key_arrays,
            t_idx=1,
        ),
        np.array([10.0, 0.0]),
    )

    inner = Vanilla(key="terminal", k=90.0, side="call")
    ko = KnockOutAnyN(inner=inner, keys=["KO_flag"])
    ki = KnockInAllN(inner=inner, keys=["KI_flag"])
    kiko = KnockInKnockOut(inner=inner, ki_key="KI_flag", ko_key="KO_flag")
    assert np.allclose(engine._evaluate_payoff_vectorized(ko, key_arrays, 1), np.array([20.0, 5.0]))
    assert np.allclose(engine._evaluate_payoff_vectorized(ki, key_arrays, 1), np.array([20.0, 5.0]))
    assert np.allclose(
        engine._evaluate_payoff_vectorized(kiko, key_arrays, 1),
        np.array([20.0, 5.0]),
    )


def test_vectorized_key_arrays_handle_barrier_time_and_lookback():
    model = BatesModel(sampler=ConstantSampler(s0=100.0), S0=100.0, r=0.0, q=0.0)
    cfg = MonteCarloConfig(n_paths=2, timegrid=np.array([0.0, 0.5, 1.0]), seed=0)
    payoff = Vanilla(key="max", k=0.0, side="call")
    features = [
        LookbackFeature(mode="both", track_time=True),
        BarrierFeature(name="B", level=102.0, direction="up", strict=False, track_time=True),
    ]
    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=European(),
        config=cfg,
    )
    paths = np.array([[100.0, 103.0, 99.0], [100.0, 101.0, 102.0]])
    key_arrays = engine._build_vectorized_key_arrays(paths)
    assert key_arrays is not None
    assert np.allclose(key_arrays["max"][:, -1], np.array([103.0, 102.0]))

    payoff_time = Vanilla(key="B_time", k=0.0, side="call")
    engine_time = MonteCarloEngine(
        model=model,
        payoff=payoff_time,
        features=features,
        exercise=European(),
        config=cfg,
    )
    key_arrays_time = engine_time._build_vectorized_key_arrays(paths)
    assert key_arrays_time is not None
    assert key_arrays_time["B_time"][0, 0] == pytest.approx(np.inf)
    assert key_arrays_time["B_time"][0, 1] == pytest.approx(0.5)


def test_fallback_path_with_custom_feature_american_and_never_exercise():
    sampler = VectorizedGBMSampler(s0=100.0, mu=0.02, sigma=0.2)
    model = BatesModel(sampler=sampler, S0=100.0, r=0.01, q=0.0)
    payoff = Vanilla(key="spread", k=0.0, side="call")
    features = [
        AsianFeature(),
        LookbackFeature(mode="both", track_time=True),
        BarrierFeature(name="B", level=120.0, direction="up", strict=False, track_time=True),
        CustomSpreadFeature(),
    ]
    cfg = MonteCarloConfig(n_paths=120, timegrid=np.linspace(0.0, 1.0, 10), seed=21)

    engine_american = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=American(),
        config=cfg,
    )
    price_a, stderr_a = engine_american.price()
    assert math.isfinite(price_a)
    assert math.isfinite(stderr_a)

    engine_never = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=NeverExercise(),
        config=cfg,
    )
    price_n, stderr_n = engine_never.price()
    assert math.isfinite(price_n)
    assert math.isfinite(stderr_n)


def test_vectorized_key_arrays_return_none_for_missing_feature_or_unknown_payoff():
    model = BatesModel(sampler=ConstantSampler(s0=100.0), S0=100.0, r=0.0, q=0.0)
    cfg = MonteCarloConfig(n_paths=2, timegrid=np.array([0.0, 1.0]), seed=0)
    engine_missing = MonteCarloEngine(
        model=model,
        payoff=Vanilla(key="terminal", k=100.0, side="call"),
        features=[],
        exercise=European(),
        config=cfg,
    )
    key_arrays = engine_missing._build_vectorized_key_arrays(
        np.array([[1.0, 2.0], [1.0, 2.0]])
    )
    assert key_arrays is not None

    class UnknownPayoff(Payoff):
        def __init__(self):
            super().__init__()
            self.requires = frozenset({"terminal"})

        def __call__(self, state: dict) -> float:
            return float(state["terminal"])

    unknown = UnknownPayoff()
    assert engine_missing._is_vectorized_payoff(unknown) is False
    with pytest.raises(TypeError, match="Unsupported vectorized payoff type"):
        engine_missing._evaluate_payoff_vectorized(
            unknown,
            {"terminal": np.array([[1.0, 2.0]])},
            t_idx=1,
        )
