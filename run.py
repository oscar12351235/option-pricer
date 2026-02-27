from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from config.run_config import RunConfig, apply_overrides, load_run_config

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


engine_mod = importlib.import_module("option-pricer.engine")
config_mod = importlib.import_module("option-pricer.config")
exercise_mod = importlib.import_module("option-pricer.exercise")
features_mod = importlib.import_module("option-pricer.features.core")
models_bates_mod = importlib.import_module("option-pricer.models.bates")
models_gbm_jump_mod = importlib.import_module("option-pricer.models.gbm_jump")
models_jumps_mod = importlib.import_module("option-pricer.models.jumps")
models_numba_mod = importlib.import_module("option-pricer.models.numba_heston_qe")
models_ql_mod = importlib.import_module("option-pricer.models.quantlib_heston")
payoffs_mod = importlib.import_module("option-pricer.payoffs.core")


MonteCarloEngine = engine_mod.MonteCarloEngine
MonteCarloConfig = config_mod.MonteCarloConfig
American = exercise_mod.American
European = exercise_mod.European
AsianFeature = features_mod.AsianFeature
BarrierFeature = features_mod.BarrierFeature
LookbackFeature = features_mod.LookbackFeature
BatesModel = models_bates_mod.BatesModel
VectorizedJumpGBMSampler = models_gbm_jump_mod.VectorizedJumpGBMSampler
NoJump = models_jumps_mod.NoJump
LognormalJumps = models_jumps_mod.LognormalJumps
NumbaHestonSampler = models_numba_mod.NumbaHestonSampler
HestonModel = models_ql_mod.HestonModel
Vanilla = payoffs_mod.Vanilla
Digital = payoffs_mod.Digital
AsianFloatingVanilla = payoffs_mod.AsianFloatingVanilla
KnockOutAnyN = payoffs_mod.KnockOutAnyN
KnockInAllN = payoffs_mod.KnockInAllN
KnockInKnockOut = payoffs_mod.KnockInKnockOut


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
        jump = NoJump()
    else:
        jump = LognormalJumps(lam=cfg.jump_lambda, mu=cfg.jump_mu, sigma=cfg.jump_sigma)
    sampler = VectorizedJumpGBMSampler(
        S0=cfg.s0,
        r=cfg.r,
        q=cfg.q,
        sigma=cfg.sigma,
        jump=jump,
    )
    return BatesModel(sampler=sampler, S0=cfg.s0, r=cfg.r, q=cfg.q)


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


def run_pricer(cfg: RunConfig) -> tuple[float, float]:
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
    return engine.price()


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the vectorized option pricer.")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
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
    overrides = {field.name: getattr(args, field.name, None) for field in fields(RunConfig)}
    cfg = apply_overrides(cfg, overrides)

    started = time.perf_counter()
    price, stderr = run_pricer(cfg)
    elapsed = time.perf_counter() - started

    print("Option pricer run")
    print(
        f"model={cfg.model_kind} exercise={cfg.exercise_kind} "
        f"payoff={cfg.payoff_kind} wrapper={cfg.wrapper_kind}"
    )
    print(f"n_paths={cfg.n_paths} n_steps={cfg.n_steps} maturity={cfg.maturity} seed={cfg.seed}")
    print(f"price={price:.10f}")
    print(f"stderr={stderr:.10f}")
    print(f"elapsed_seconds={elapsed:.4f}")


if __name__ == "__main__":
    main()
