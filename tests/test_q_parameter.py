from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from option_pricer.models.gbm_jump import VectorizedJumpGBMSampler
from option_pricer.models.jumps import NoJump


def test_gbm_sampler_q_changes_drift_deterministically_when_sigma_is_zero():
    timegrid = np.array([0.0, 0.5, 1.0], dtype=float)
    zeros = np.zeros((len(timegrid) - 1, 3, 2), dtype=float)

    low_q_sampler = VectorizedJumpGBMSampler(S0=100.0, r=0.03, q=0.00, sigma=0.0, jump=NoJump())
    high_q_sampler = VectorizedJumpGBMSampler(S0=100.0, r=0.03, q=0.05, sigma=0.0, jump=NoJump())

    rng = np.random.default_rng(7)
    low_q_paths = low_q_sampler(n_paths=3, timegrid=timegrid, randnumbergenerator=rng, z=zeros)
    high_q_paths = high_q_sampler(n_paths=3, timegrid=timegrid, randnumbergenerator=rng, z=zeros)

    expected_low_terminal = 100.0 * math.exp((0.03 - 0.00) * 1.0)
    expected_high_terminal = 100.0 * math.exp((0.03 - 0.05) * 1.0)

    assert np.allclose(low_q_paths[:, -1], expected_low_terminal)
    assert np.allclose(high_q_paths[:, -1], expected_high_terminal)
    assert np.all(high_q_paths[:, -1] < low_q_paths[:, -1])
