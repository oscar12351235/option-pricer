from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

import run


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


def test_graph_generation_is_separate_and_writes_file(tmp_path: Path):
    cfg = run.RunConfig(
        model_kind="gbm_jump",
        exercise_kind="american",
        payoff_kind="vanilla",
        wrapper_kind="knockin",
        jump_kind="none",
        n_paths=120,
        n_steps=12,
        seed=5,
        use_crn=True,
        antithetic=True,
    )
    price, stderr, engine = run.run_pricer_with_engine(cfg)
    assert math.isfinite(price)
    assert math.isfinite(stderr)

    output_path = tmp_path / "paths.png"
    graph_path = run.generate_paths_graph(
        engine=engine,
        output_path=output_path,
        max_paths=30,
    )
    assert graph_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.parametrize(
    "model_kind, jump_kind, n_factors",
    [("numba_heston", "none", 2), ("numba_heston", "lognormal", 4), ("gbm_jump", "lognormal", 2)],
)
def test_build_engine_selects_model_and_jumps(model_kind, jump_kind, n_factors):
    cfg = run.RunConfig(model_kind=model_kind, jump_kind=jump_kind, n_paths=200, n_steps=8)
    engine = run.build_engine(cfg)
    assert engine.model.n_factors == n_factors
    price, stderr = engine.price()
    assert math.isfinite(price) and price >= 0.0
    assert math.isfinite(stderr) and stderr >= 0.0


def test_quantlib_heston_pipeline_runs():
    cfg = run.RunConfig(
        model_kind="quantlib_heston",
        exercise_kind="european",
        payoff_kind="vanilla",
        wrapper_kind="none",
        n_paths=200,
        n_steps=8,
        antithetic=False,
    )
    price, stderr = run.run_pricer(cfg)
    assert math.isfinite(price) and price > 0.0
    assert stderr > 0.0


def test_cli_prints_results_and_writes_plot(tmp_path: Path, monkeypatch, capsys):
    plot_file = tmp_path / "paths.png"
    argv = [
        "run.py",
        "--config-dir",
        str(Path(run.__file__).resolve().parent / "config"),
        "--model",
        "numba_heston",
        "--jump-kind",
        "lognormal",
        "--exercise",
        "american",
        "--n-paths",
        "400",
        "--n-steps",
        "12",
        "--plot",
        "--plot-file",
        str(plot_file),
        "--plot-max-paths",
        "20",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()

    out = capsys.readouterr().out
    assert "model=numba_heston exercise=american" in out
    assert "price=" in out and "stderr=" in out
    assert f"graph_file={plot_file}" in out
    assert plot_file.stat().st_size > 0


def test_cli_without_plot_does_not_write_graph(tmp_path: Path, monkeypatch, capsys):
    plot_file = tmp_path / "paths.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--n-paths",
            "200",
            "--n-steps",
            "6",
            "--no-plot",
            "--plot-file",
            str(plot_file),
        ],
    )
    run.main()
    assert "price=" in capsys.readouterr().out
    assert not plot_file.exists()
