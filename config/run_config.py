from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

MODEL_KINDS = {"numba_heston", "quantlib_heston", "gbm_jump"}
EXERCISE_KINDS = {"european", "american"}
PAYOFF_KINDS = {"vanilla", "digital", "asian_floating"}
WRAPPER_KINDS = {"none", "knockin", "knockout", "kiko"}
DIGITAL_VARIANTS = {"cash", "asset"}
JUMP_KINDS = {"none", "lognormal"}


@dataclass(frozen=True)
class RunConfig:
    model_kind: str = "numba_heston"
    exercise_kind: str = "american"
    payoff_kind: str = "vanilla"
    wrapper_kind: str = "knockin"
    digital_variant: str = "cash"
    jump_kind: str = "none"
    n_paths: int = 20_000
    n_steps: int = 252
    maturity: float = 1.0
    seed: int = 42
    use_crn: bool = True
    antithetic: bool = True
    s0: float = 100.0
    strike: float = 100.0
    r: float = 0.01
    q: float = 0.00
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.07
    sigma: float = 0.4
    rho: float = -0.7
    barrier_ki: float = 120.0
    barrier_ko: float = 80.0
    digital_cash: float = 1.0
    jump_lambda: float = 0.30
    jump_mu: float = -0.10
    jump_sigma: float = 0.25


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid TOML object in {path}: expected table")
    return data


def _section(data: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = data.get(section_name, data)
    if not isinstance(section, dict):
        raise ValueError(f"Section '{section_name}' must be a table")
    return dict(section)


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        allowed_txt = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {name}={value!r}; expected one of: {allowed_txt}")


def _validate(cfg: RunConfig) -> None:
    _validate_choice("model_kind", cfg.model_kind, MODEL_KINDS)
    _validate_choice("exercise_kind", cfg.exercise_kind, EXERCISE_KINDS)
    _validate_choice("payoff_kind", cfg.payoff_kind, PAYOFF_KINDS)
    _validate_choice("wrapper_kind", cfg.wrapper_kind, WRAPPER_KINDS)
    _validate_choice("digital_variant", cfg.digital_variant, DIGITAL_VARIANTS)
    _validate_choice("jump_kind", cfg.jump_kind, JUMP_KINDS)

    if cfg.n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.maturity <= 0:
        raise ValueError("maturity must be positive")
    if cfg.antithetic and cfg.n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic is enabled")


def apply_overrides(base: RunConfig, overrides: Mapping[str, Any]) -> RunConfig:
    field_names = {field.name for field in fields(RunConfig)}
    cleaned: dict[str, Any] = {}

    for key, value in overrides.items():
        if value is None:
            continue
        if key not in field_names:
            raise ValueError(f"Unknown run config key: {key}")
        cleaned[key] = value

    cfg = replace(base, **cleaned)
    _validate(cfg)
    return cfg


def load_run_config(config_dir: Path | str = "config") -> RunConfig:
    config_dir = Path(config_dir)

    config_part = _section(_read_toml(config_dir / "config.toml"), "run")
    return apply_overrides(RunConfig(), config_part)
