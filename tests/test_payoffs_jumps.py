from __future__ import annotations

import math

import numpy as np
import pytest

from option_pricer.models import VectorizedJumpGBMSampler
from option_pricer.models.bates import BatesModel
from option_pricer.models.jumps import LognormalJumps, NoJump
from option_pricer.payoffs.core import (
    AsianFloatingVanilla,
    Digital,
    KnockInAllN,
    KnockInKnockOut,
    KnockOutAnyN,
    Vanilla,
)


def test_payoff_variants_and_wrappers():
    with pytest.raises(ValueError, match="call' or 'put"):
        Vanilla(key="terminal", k=100.0, side="bad")
    with pytest.raises(ValueError, match="call' or 'put"):
        Digital(key="terminal", k=100.0, side="bad", cash=1.0, variant="cash")
    with pytest.raises(ValueError, match="cash' or 'asset"):
        Digital(key="terminal", k=100.0, side="call", cash=1.0, variant="bad")
    with pytest.raises(ValueError, match="call' or 'put"):
        AsianFloatingVanilla(key1="terminal", key2="average", side="bad")

    state = {
        "terminal": 110.0,
        "average": 105.0,
        "KI_flag": True,
        "KO_flag": False,
    }

    d_call_cash = Digital(key="terminal", k=100.0, side="call", cash=2.5, variant="cash")
    d_put_asset = Digital(key="terminal", k=120.0, side="put", cash=1.0, variant="asset")
    asian_put = AsianFloatingVanilla(key1="terminal", key2="average", side="put")
    assert d_call_cash(state) == pytest.approx(2.5)
    assert d_put_asset(state) == pytest.approx(110.0)
    assert asian_put(state) == pytest.approx(0.0)

    inner = Vanilla(key="terminal", k=100.0, side="call")
    ko = KnockOutAnyN(inner=inner, keys=["KO_flag"])
    ki = KnockInAllN(inner=inner, keys=["KI_flag"])
    kiko = KnockInKnockOut(inner=inner, ki_key="KI_flag", ko_key="KO_flag")
    assert ko(state) == pytest.approx(10.0)
    assert ki(state) == pytest.approx(10.0)
    assert kiko(state) == pytest.approx(10.0)

    assert ko({**state, "KO_flag": True}) == pytest.approx(0.0)
    assert ki({**state, "KI_flag": False}) == pytest.approx(0.0)
    assert kiko({**state, "KI_flag": False}) == pytest.approx(0.0)
    assert kiko({**state, "KO_flag": True}) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="different"):
        KnockInKnockOut(inner=inner, ki_key="X", ko_key="X")


class FakeRng:
    def __init__(self, N, z):
        self._N = np.asarray(N, dtype=int)
        self._z = np.asarray(z, dtype=float)

    def poisson(self, lam, size):
        assert size == self._N.shape[0]
        return self._N

    def standard_normal(self, size):
        return self._z[:size]


def test_jump_models_cover_shape_paths_and_error_branch():
    no_jump = NoJump()
    assert no_jump.compensator() == pytest.approx(0.0)
    assert np.allclose(no_jump.step_multiplier(4, 0.1, randnumbergenerator=None), np.ones(4))

    jump = LognormalJumps(lam=0.5, mu=-0.1, sigma=0.2)
    expected_k = np.exp(-0.1 + 0.5 * 0.2 * 0.2) - 1.0
    assert jump.compensator() == pytest.approx(expected_k)

    # Branch where no jumps happen (idx.size == 0)
    rng_none = FakeRng(N=[0, 0, 0], z=[0.0, 0.0, 0.0])
    out_none = jump.step_multiplier(3, 0.2, rng_none, z=np.array([0.1, 0.2, 0.3]))
    assert np.allclose(out_none, np.ones(3))

    # Branch where z has full n length and is filtered by idx
    rng_some = FakeRng(N=[0, 2, 1, 0], z=[0.5, -0.1, 0.3, 0.0])
    out_full = jump.step_multiplier(4, 0.2, rng_some, z=np.array([0.7, 0.2, -0.4, 0.1]))
    assert out_full.shape == (4,)
    assert out_full[0] == pytest.approx(1.0)
    assert out_full[3] == pytest.approx(1.0)

    # Branch where z has only number-of-jumps length
    out_short = jump.step_multiplier(4, 0.2, rng_some, z=np.array([0.2, -0.1]))
    assert out_short.shape == (4,)

    with pytest.raises(ValueError, match="must have length n"):
        jump.step_multiplier(4, 0.2, rng_some, z=np.array([0.1]))


@pytest.mark.parametrize(
    "payoff, expected",
    [
        (Vanilla(key="terminal", k=120.0, side="put"), 10.0),
        (Vanilla(key="average", k=100.0, side="call"), 5.0),
        (Digital(key="terminal", k=100.0, side="call", cash=1.0, variant="asset"), 110.0),
        (Digital(key="terminal", k=120.0, side="call", cash=1.0, variant="cash"), 0.0),
        (Digital(key="average", k=110.0, side="put", cash=3.0, variant="cash"), 3.0),
        (Digital(key="terminal", k=100.0, side="put", cash=1.0, variant="cash"), 0.0),
        (AsianFloatingVanilla(key1="terminal", key2="average", side="call"), 5.0),
    ],
)
def test_payoff_values(payoff, expected):
    state = {"terminal": 110.0, "average": 105.0}
    assert payoff(state) == pytest.approx(expected)
    assert payoff.requires <= set(state)


def test_wrappers_combine_requirements_and_need_all_knock_in_flags():
    inner = Vanilla(key="average", k=100.0, side="call")
    double_ki = KnockInAllN(inner=inner, keys=["A_flag", "B_flag"])
    double_ko = KnockOutAnyN(inner=inner, keys=["A_flag", "B_flag", "A_flag"])
    assert double_ki.requires == {"average", "A_flag", "B_flag"}
    assert double_ko.requires == {"average", "A_flag", "B_flag"}

    state = {"average": 104.0, "A_flag": True, "B_flag": False}
    assert double_ki(state) == 0.0
    assert double_ki({**state, "B_flag": True}) == pytest.approx(4.0)
    assert double_ko(state) == 0.0
    assert double_ko({**state, "A_flag": False}) == pytest.approx(4.0)


def test_lognormal_jumps_draw_sizes_from_generator_when_z_is_omitted():
    jump = LognormalJumps(lam=50.0, mu=-0.1, sigma=0.2)
    multipliers = jump.step_multiplier(100_000, 0.1, np.random.default_rng(0))
    # E[prod e^J over one step] = exp(lam * dt * k), with k the compensator.
    expected = math.exp(50.0 * 0.1 * jump.compensator())
    assert np.mean(multipliers) == pytest.approx(expected, rel=0.02)
    assert np.all(multipliers > 0.0)


@pytest.mark.parametrize("jump", [NoJump(), LognormalJumps(lam=0.5, mu=-0.1, sigma=0.25)])
def test_gbm_jump_sampler_is_a_martingale_after_discounting(jump):
    r, q, t = 0.04, 0.01, 1.0
    sampler = VectorizedJumpGBMSampler(S0=100.0, r=r, q=q, sigma=0.2, jump=jump)
    paths = sampler(100_000, np.linspace(0.0, t, 5), np.random.default_rng(1))
    discounted = math.exp(-r * t) * paths[:, -1]
    stderr = discounted.std(ddof=1) / math.sqrt(len(discounted))
    assert abs(discounted.mean() - 100.0 * math.exp(-q * t)) <= 4 * stderr


def test_flat_rate_model_wrapper():
    sampler = VectorizedJumpGBMSampler(S0=95.0, r=0.05, q=0.0, sigma=0.2, jump=NoJump())
    model = BatesModel(sampler=sampler, S0=95.0, r=0.05, q=0.0)
    assert model.s0() == 95.0
    assert model.discount(2.0) == pytest.approx(math.exp(-0.1))
    assert model.path_sampler() is sampler
    assert model.supports_crn and model.supports_antithetic
    assert model.n_factors == 2
