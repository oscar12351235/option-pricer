from __future__ import annotations

import math

import numpy as np
import pytest

from option_pricer import (
    AsianFeature,
    BarrierFeature,
    BatesModel,
    European,
    LookbackFeature,
    MonteCarloConfig,
    MonteCarloEngine,
    Vanilla,
)

GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
PATH = np.array([100.0, 104.0, 97.0, 110.0, 103.0])


def test_asian_average_includes_initial_point():
    state: dict = {}
    AsianFeature().apply(PATH, state)
    assert AsianFeature().provides == frozenset({"average"})
    assert state["average"] == pytest.approx(PATH.mean())


@pytest.mark.parametrize(
    "direction, level, strict, expected_time",
    [
        ("up", 104.0, False, 0.25),  # touches 104 at t = 0.25
        ("up", 104.0, True, 0.75),  # strictly above 104 only at t = 0.75
        ("down", 97.0, False, 0.5),
        ("down", 97.0, True, math.inf),  # never strictly below 97
        ("up", 120.0, False, math.inf),
    ],
)
def test_barrier_flag_and_first_hitting_time(direction, level, strict, expected_time):
    feature = BarrierFeature(
        name="B", level=level, direction=direction, strict=strict, track_time=True
    )
    state: dict = {"t_grid": GRID}
    feature.apply(PATH, state)

    assert feature.provides == frozenset({"B_flag", "B_time"})
    assert state["B_flag"] is (expected_time != math.inf)
    assert state["B_time"] == expected_time


def test_barrier_without_time_tracking_ignores_time_grid():
    feature = BarrierFeature(
        name="KO", level=98.0, direction="down", strict=False, track_time=False
    )
    state: dict = {}
    feature.apply(PATH, state)
    assert feature.provides == frozenset({"KO_flag"})
    assert state == {"KO_flag": True}


def test_lookback_extrema_and_their_first_times():
    feature = LookbackFeature(mode="both", track_time=True)
    state: dict = {"t_grid": GRID}
    feature.apply(np.array([100.0, 110.0, 95.0, 110.0, 95.0]), state)

    assert feature.provides == frozenset({"max", "min", "max_time", "min_time"})
    assert {k: state[k] for k in feature.provides} == {
        "max": 110.0,
        "max_time": 0.25,
        "min": 95.0,
        "min_time": 0.5,
    }


@pytest.mark.parametrize(
    "mode, keys", [("max", {"max"}), ("min", {"min"}), ("both", {"max", "min"})]
)
def test_lookback_modes_without_time(mode, keys):
    feature = LookbackFeature(mode=mode, track_time=False)
    state: dict = {}
    feature.apply(PATH, state)
    assert feature.provides == frozenset(keys)
    assert set(state) == keys


def test_invalid_feature_arguments_are_rejected():
    with pytest.raises(ValueError, match="'up' or 'down'"):
        BarrierFeature(name="B", level=1.0, direction="sideways", strict=False, track_time=False)
    with pytest.raises(ValueError, match="'max', 'min' or 'both'"):
        LookbackFeature(mode="median", track_time=False)


@pytest.mark.parametrize(
    "feature",
    [
        LookbackFeature(mode="max", track_time=True),
        BarrierFeature(name="B", level=104.0, direction="up", strict=False, track_time=True),
    ],
)
def test_time_tracking_requires_a_matching_time_grid(feature):
    with pytest.raises(ValueError, match="must be a list"):
        feature.apply(PATH, {})
    with pytest.raises(ValueError, match="length must equal"):
        feature.apply(PATH, {"t_grid": GRID[:-1]})


def test_vectorised_feature_arrays_match_reference_apply():
    """The engine's array implementation agrees with Feature.apply on every path."""
    rng = np.random.default_rng(0)
    timegrid = np.linspace(0.0, 1.0, 30)
    paths = 100.0 * np.exp(np.cumsum(0.05 * rng.standard_normal((200, 30)), axis=1))
    paths[:, 0] = 100.0
    features = [
        AsianFeature(),
        LookbackFeature(mode="both", track_time=True),
        BarrierFeature(name="UP", level=105.0, direction="up", strict=True, track_time=True),
        BarrierFeature(name="DN", level=95.0, direction="down", strict=False, track_time=True),
    ]
    keys = sorted(set().union(*(f.provides for f in features)))

    class Sampler:
        def __call__(self, n_paths, timegrid, randnumbergenerator, randnumber_key=None, z=None):
            return paths

    class AllKeys(Vanilla):
        def __init__(self):
            super().__init__(key="terminal", k=0.0, side="call")
            self.requires = frozenset(keys)

    engine = MonteCarloEngine(
        model=BatesModel(sampler=Sampler(), S0=100.0, r=0.0, q=0.0),
        payoff=AllKeys(),
        features=features,
        exercise=European(),
        config=MonteCarloConfig(n_paths=200, timegrid=timegrid, seed=0),
    )
    arrays = engine._build_vectorized_key_arrays(paths)
    assert arrays is not None

    for i in range(len(paths)):
        state: dict = {"t_grid": timegrid}
        for feature in features:
            feature.apply(paths[i], state)
        for key in keys:
            assert arrays[key][i, -1] == pytest.approx(state[key], rel=1e-12), key
