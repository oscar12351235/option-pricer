from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from config.run_config import RunConfig, apply_overrides, load_run_config

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from option_pricer.config import MonteCarloConfig
from option_pricer.engine import MonteCarloEngine
from option_pricer.exercise import American, European
from option_pricer.features.core import AsianFeature, BarrierFeature, LookbackFeature
from option_pricer.models.bates import BatesModel
from option_pricer.models.gbm_jump import VectorizedJumpGBMSampler
from option_pricer.models.jumps import Jump, LognormalJumps, NoJump
from option_pricer.models.numba_heston_qe import NumbaHestonSampler
from option_pricer.models.quantlib_heston import HestonModel
from option_pricer.payoffs.core import (
    AsianFloatingVanilla,
    Digital,
    KnockInAllN,
    KnockInKnockOut,
    KnockOutAnyN,
    Payoff,
    Vanilla,
)


def _build_model(cfg: RunConfig):
    if cfg.model_kind == "numba_heston":
        sampler = NumbaHestonSampler(
            S0=cfg.s0,
            v0=cfg.v0,
            kappa=cfg.kappa,
            theta=cfg.theta,
            sigma=cfg.sigma,
            rho=cfg.rho,
            r=cfg.r,
            q=cfg.q,
            v_floor=1e-12,
        )
        return BatesModel(sampler=sampler, S0=cfg.s0, r=cfg.r, q=cfg.q)

    if cfg.model_kind == "quantlib_heston":
        import QuantLib as ql

        calendar = ql.TARGET()
        day_count = ql.Actual365Fixed()
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        s0_handle = ql.QuoteHandle(ql.SimpleQuote(cfg.s0))
        risk_free_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, cfg.r, day_count, ql.Continuous)
        )
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, cfg.q, day_count, ql.Continuous)
        )
        _ = calendar

        return HestonModel(
            s0_handle=s0_handle,
            riskFreeTS=risk_free_ts,
            dividendTS=dividend_ts,
            v0=cfg.v0,
            kappa=cfg.kappa,
            theta=cfg.theta,
            sigma=cfg.sigma,
            rho=cfg.rho,
        )

    if cfg.jump_kind == "none":
        jump: Jump = NoJump()
    else:
        jump = LognormalJumps(lam=cfg.jump_lambda, mu=cfg.jump_mu, sigma=cfg.jump_sigma)
    jump_sampler = VectorizedJumpGBMSampler(
        S0=cfg.s0,
        r=cfg.r,
        q=cfg.q,
        sigma=cfg.sigma,
        jump=jump,
    )
    return BatesModel(sampler=jump_sampler, S0=cfg.s0, r=cfg.r, q=cfg.q)


def _build_features(cfg: RunConfig):
    return [
        AsianFeature(),
        BarrierFeature(
            name="KI",
            level=cfg.barrier_ki,
            direction="up",
            strict=False,
            track_time=True,
        ),
        BarrierFeature(
            name="KO",
            level=cfg.barrier_ko,
            direction="down",
            strict=False,
            track_time=True,
        ),
        LookbackFeature(mode="both", track_time=True),
    ]


def _build_payoff(cfg: RunConfig):
    base: Payoff
    if cfg.payoff_kind == "vanilla":
        base = Vanilla(key="average", k=cfg.strike, side="call")
    elif cfg.payoff_kind == "digital":
        base = Digital(
            key="average",
            k=cfg.strike,
            side="call",
            cash=cfg.digital_cash,
            variant=cfg.digital_variant,
        )
    else:
        base = AsianFloatingVanilla(key1="terminal", key2="average", side="call")

    if cfg.wrapper_kind == "none":
        return base
    if cfg.wrapper_kind == "knockin":
        return KnockInAllN(inner=base, keys=["KI_flag"])
    if cfg.wrapper_kind == "knockout":
        return KnockOutAnyN(inner=base, keys=["KO_flag"])
    return KnockInKnockOut(inner=base, ki_key="KI_flag", ko_key="KO_flag")


def _build_exercise(cfg: RunConfig):
    if cfg.exercise_kind == "european":
        return European()
    return American()


def build_engine(cfg: RunConfig):
    timegrid = np.linspace(0.0, cfg.maturity, cfg.n_steps + 1)
    mc_cfg = MonteCarloConfig(
        n_paths=cfg.n_paths,
        timegrid=timegrid,
        seed=cfg.seed,
        use_crn=cfg.use_crn,
        antithetic=cfg.antithetic,
    )

    model = _build_model(cfg)
    payoff = _build_payoff(cfg)
    features = _build_features(cfg)
    exercise = _build_exercise(cfg)

    engine = MonteCarloEngine(
        model=model,
        payoff=payoff,
        features=features,
        exercise=exercise,
        config=mc_cfg,
    )
    return engine


def run_pricer_with_engine(cfg: RunConfig) -> tuple[float, float, MonteCarloEngine]:
    engine = build_engine(cfg)
    price, stderr = engine.price()
    return price, stderr, engine


def run_pricer(cfg: RunConfig) -> tuple[float, float]:
    price, stderr, _ = run_pricer_with_engine(cfg)
    return price, stderr


def generate_paths_graph(
    engine: MonteCarloEngine,
    output_path: Path,
    max_paths: int = 5000,
    randnumber_key: str = "graph",
) -> Path:
    mpl_cache_dir = PROJECT_ROOT / ".matplotlib-cache"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = engine._simulate_paths(randnumber_key=randnumber_key)
    n_plot = min(int(max_paths), paths.shape[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(engine.timegrid, paths[:n_plot, :].T, alpha=0.12, linewidth=0.9)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Underlying price")
    ax.set_title(f"Simulated paths ({n_plot} of {paths.shape[0]})")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    return output_path


def print_pricing_results(cfg: RunConfig, price: float, stderr: float, elapsed: float) -> None:
    print("Option pricer run")
    print(
        f"model={cfg.model_kind} exercise={cfg.exercise_kind} "
        f"payoff={cfg.payoff_kind} wrapper={cfg.wrapper_kind}"
    )
    print(f"n_paths={cfg.n_paths} n_steps={cfg.n_steps} maturity={cfg.maturity} seed={cfg.seed}")
    print(f"price={price:.10f}")
    print(f"stderr={stderr:.10f}")
    print(f"elapsed_seconds={elapsed:.4f}")
    sys.stdout.flush()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vectorized option pricer.")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--plot", dest="plot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-file", type=Path, default=Path("outputs/last_run_paths.png"))
    parser.add_argument("--plot-max-paths", type=int, default=5000)
    parser.add_argument(
        "--model",
        dest="model_kind",
        choices=["numba_heston", "quantlib_heston", "gbm_jump"],
        default=None,
    )
    parser.add_argument(
        "--exercise",
        dest="exercise_kind",
        choices=["european", "american"],
        default=None,
    )
    parser.add_argument(
        "--payoff-kind",
        dest="payoff_kind",
        choices=["vanilla", "digital", "asian_floating"],
        default=None,
    )
    parser.add_argument(
        "--wrapper",
        dest="wrapper_kind",
        choices=["none", "knockin", "knockout", "kiko"],
        default=None,
    )
    parser.add_argument(
        "--digital-variant",
        dest="digital_variant",
        choices=["cash", "asset"],
        default=None,
    )
    parser.add_argument(
        "--jump-kind",
        dest="jump_kind",
        choices=["none", "lognormal"],
        default=None,
    )
    parser.add_argument("--n-paths", dest="n_paths", type=int, default=None)
    parser.add_argument("--n-steps", dest="n_steps", type=int, default=None)
    parser.add_argument("--maturity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use-crn",
        dest="use_crn",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--antithetic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--s0", type=float, default=None)
    parser.add_argument("--strike", type=float, default=None)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--v0", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--barrier-ki", dest="barrier_ki", type=float, default=None)
    parser.add_argument("--barrier-ko", dest="barrier_ko", type=float, default=None)
    parser.add_argument("--digital-cash", dest="digital_cash", type=float, default=None)
    parser.add_argument("--jump-lambda", dest="jump_lambda", type=float, default=None)
    parser.add_argument("--jump-mu", dest="jump_mu", type=float, default=None)
    parser.add_argument("--jump-sigma", dest="jump_sigma", type=float, default=None)
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    cfg = load_run_config(args.config_dir)
    overrides = {
        field_name: getattr(args, field_name, None) for field_name in RunConfig.model_fields
    }
    cfg = apply_overrides(cfg, overrides)

    started = time.perf_counter()
    price, stderr, engine = run_pricer_with_engine(cfg)
    elapsed = time.perf_counter() - started

    print_pricing_results(cfg, price, stderr, elapsed)

    if not cfg.plot:
        return

    try:
        plot_started = time.perf_counter()
        graph_path = generate_paths_graph(
            engine=engine,
            output_path=args.plot_file,
            max_paths=args.plot_max_paths,
        )
        plot_elapsed = time.perf_counter() - plot_started
        print(f"graph_file={graph_path}")
        print(f"graph_elapsed_seconds={plot_elapsed:.4f}")
    except Exception as exc:  # pragma: no cover - best-effort graphing after pricing
        print(f"graph_error={exc}")


if __name__ == "__main__":
    main()
