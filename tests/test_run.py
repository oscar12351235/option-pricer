from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
run = importlib.import_module("run")


def test_run_pricer_gbm_jump_pipeline_smoke():
    cfg = run.RunConfig(
        model_kind="gbm_jump",
        exercise_kind="american",
        payoff_kind="digital",
        wrapper_kind="kiko",
        jump_kind="lognormal",
        n_paths=400,
        n_steps=24,
        seed=19,
        use_crn=True,
        antithetic=True,
    )

    price, stderr = run.run_pricer(cfg)

    assert math.isfinite(price)
    assert math.isfinite(stderr)
    assert price >= 0.0
    assert stderr >= 0.0
