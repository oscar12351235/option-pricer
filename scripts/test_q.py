from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

run = importlib.import_module("run")


def main() -> int:
    q_values = [0.00, 0.02, 0.05]
    base_cfg = run.RunConfig(
        model_kind="gbm_jump",
        exercise_kind="european",
        payoff_kind="vanilla",
        wrapper_kind="none",
        jump_kind="none",
        plot=False,
        n_paths=20_000,
        n_steps=64,
        maturity=1.0,
        seed=42,
        use_crn=True,
        antithetic=True,
        s0=100.0,
        strike=100.0,
        r=0.03,
        sigma=0.20,
    )

    results: list[tuple[float, float, float]] = []
    for q_value in q_values:
        cfg = run.apply_overrides(base_cfg, {"q": q_value})
        price, stderr = run.run_pricer(cfg)
        results.append((q_value, price, stderr))

    print("Testing q as dividend yield in the option pricer")
    print("q\tprice\tstderr")
    for q_value, price, stderr in results:
        print(f"{q_value:.4f}\t{price:.6f}\t{stderr:.6f}")

    is_monotone = all(
        earlier[1] >= later[1] for earlier, later in zip(results, results[1:], strict=False)
    )
    print(f"monotone_nonincreasing={is_monotone}")

    return 0 if is_monotone else 1


if __name__ == "__main__":
    raise SystemExit(main())
