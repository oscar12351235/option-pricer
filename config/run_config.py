from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_kind: Literal["numba_heston", "quantlib_heston", "gbm_jump"] = "numba_heston"
    exercise_kind: Literal["european", "american"] = "american"
    payoff_kind: Literal["vanilla", "digital", "asian_floating"] = "vanilla"
    wrapper_kind: Literal["none", "knockin", "knockout", "kiko"] = "knockin"
    digital_variant: Literal["cash", "asset"] = "cash"
    jump_kind: Literal["none", "lognormal"] = "none"
    plot: bool = True
    n_paths: int = Field(default=20_000, gt=0)
    n_steps: int = Field(default=252, gt=0)
    maturity: float = Field(default=1.0, gt=0.0)
    seed: int = Field(default=42, ge=0)
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

    @model_validator(mode="after")
    def _validate_antithetic_paths(self) -> RunConfig:
        if self.antithetic and self.n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic is enabled")
        return self


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


def apply_overrides(base: RunConfig, overrides: Mapping[str, Any]) -> RunConfig:
    field_names = set(RunConfig.model_fields)
    cleaned: dict[str, Any] = {}

    for key, value in overrides.items():
        if value is None:
            continue
        if key not in field_names:
            raise ValueError(f"Unknown run config key: {key}")
        cleaned[key] = value

    merged = base.model_dump()
    merged.update(cleaned)
    try:
        return RunConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def load_run_config(config_dir: Path | str = "config") -> RunConfig:
    config_dir = Path(config_dir)

    config_part = _section(_read_toml(config_dir / "config.toml"), "run")
    return apply_overrides(RunConfig(), config_part)
