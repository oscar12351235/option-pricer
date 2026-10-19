from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.run_config import RunConfig, apply_overrides, load_run_config


def test_load_run_config_from_toml_stack(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[run]",
                'model_kind = "gbm_jump"',
                'exercise_kind = "european"',
                'payoff_kind = "digital"',
                'wrapper_kind = "none"',
                'digital_variant = "cash"',
                'jump_kind = "lognormal"',
                "n_paths = 400",
                "n_steps = 20",
                "maturity = 0.5",
                "seed = 17",
                "use_crn = true",
                "antithetic = true",
                "s0 = 98.0",
                "strike = 95.0",
                "r = 0.02",
                "q = 0.01",
                "v0 = 0.03",
                "kappa = 1.2",
                "theta = 0.06",
                "sigma = 0.25",
                "rho = -0.4",
                "barrier_ki = 130.0",
                "barrier_ko = 70.0",
                "digital_cash = 1.5",
                "jump_lambda = 0.4",
                "jump_mu = -0.05",
                "jump_sigma = 0.3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_run_config(tmp_path)

    assert cfg.model_kind == "gbm_jump"
    assert cfg.exercise_kind == "european"
    assert cfg.n_paths == 400
    assert cfg.n_steps == 20
    assert cfg.s0 == pytest.approx(98.0)
    assert cfg.jump_sigma == pytest.approx(0.3)


def test_apply_overrides_validates_antithetic_even_paths():
    base = RunConfig()
    updated = apply_overrides(base, {"n_paths": 600, "antithetic": True})
    assert updated.n_paths == 600
    assert updated.antithetic is True

    with pytest.raises(ValueError, match="even"):
        apply_overrides(base, {"n_paths": 601, "antithetic": True})
