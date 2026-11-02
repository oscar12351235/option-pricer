from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import QuantLib as ql

from option_pricer import European, HestonModel, MonteCarloConfig, MonteCarloEngine, Vanilla

S0, R, Q = 100.0, 0.03, 0.01
PARAMS = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)


@pytest.fixture(scope="module")
def curves():
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    return dict(
        today=today,
        s0_handle=ql.QuoteHandle(ql.SimpleQuote(S0)),
        riskFreeTS=ql.YieldTermStructureHandle(ql.FlatForward(today, R, day_count, ql.Continuous)),
        dividendTS=ql.YieldTermStructureHandle(ql.FlatForward(today, Q, day_count, ql.Continuous)),
    )


def make_model(curves, **overrides) -> HestonModel:
    kwargs = {k: curves[k] for k in ("s0_handle", "riskFreeTS", "dividendTS")}
    return HestonModel(**kwargs, **{**PARAMS, **overrides})


def test_spot_and_discount_come_from_the_curves(curves):
    model = make_model(curves)
    assert model.s0() == pytest.approx(S0)
    assert model.discount(0.0) == pytest.approx(1.0)
    assert model.discount(2.0) == pytest.approx(math.exp(-2.0 * R))


@pytest.mark.parametrize(
    "override",
    [{"rho": -1.2}, {"v0": -0.01}, {"theta": -0.01}, {"sigma": -0.1}, {"kappa": 0.0}],
)
def test_invalid_parameters_are_rejected(curves, override):
    with pytest.raises(ValueError):
        make_model(curves, **override)


def test_feller_violation_warns_but_is_allowed(curves):
    with pytest.warns(UserWarning, match="Feller"):
        make_model(curves, sigma=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_model(curves)  # 2 * 1.5 * 0.04 = 0.12 >= 0.09: no warning


def test_sampler_shape_positivity_and_reproducibility(curves):
    sampler = make_model(curves).path_sampler()
    grid = np.linspace(0.0, 1.0, 13)
    a = sampler(200, grid, np.random.default_rng(1))
    b = sampler(200, grid, np.random.default_rng(1))
    c = sampler(200, grid, np.random.default_rng(2))

    assert a.shape == (200, 13)
    assert np.all(a[:, 0] == S0)
    assert np.all(a > 0.0)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_european_call_agrees_with_analytic_heston(curves):
    model = make_model(curves)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ql.EuropeanExercise(curves["today"] + 365)
    )
    option.setPricingEngine(ql.AnalyticHestonEngine(model._ql_model))
    reference = option.NPV()

    engine = MonteCarloEngine(
        model=model,
        payoff=Vanilla(key="terminal", k=100.0, side="call"),
        features=[],
        exercise=European(),
        config=MonteCarloConfig(n_paths=8_000, timegrid=np.linspace(0.0, 1.0, 26), seed=5),
    )
    price, stderr = engine.price()
    assert abs(price - reference) <= 3.0 * stderr + 1e-3 * reference
