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
AsianFeature = features_mod.AsianFeature
BarrierFeature = features_mod.BarrierFeature
BatesModel = models_mod.BatesModel
KnockInAllN = payoffs_mod.KnockInAllN
Vanilla = payoffs_mod.Vanilla


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
