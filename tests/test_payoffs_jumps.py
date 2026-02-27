from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

payoffs_mod = importlib.import_module("option-pricer.payoffs.core")
jumps_mod = importlib.import_module("option-pricer.models.jumps")

AsianFloatingVanilla = payoffs_mod.AsianFloatingVanilla
Digital = payoffs_mod.Digital
KnockInAllN = payoffs_mod.KnockInAllN
KnockInKnockOut = payoffs_mod.KnockInKnockOut
KnockOutAnyN = payoffs_mod.KnockOutAnyN
Vanilla = payoffs_mod.Vanilla
LognormalJumps = jumps_mod.LognormalJumps
NoJump = jumps_mod.NoJump


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
