"""Payoffs evaluated on a per-path state dictionary.

A payoff declares the state keys it reads in ``requires`` (for example
``"terminal"``, ``"average"``, ``"max"`` or ``"KI_flag"``). The keys other than
``"terminal"`` are produced by :mod:`option_pricer.features`, so the same
payoff class covers several products: ``Vanilla(key="average")`` is a
fixed-strike Asian option and ``Vanilla(key="max")`` a fixed-strike lookback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any


class Payoff(ABC):
    """Base class: maps a path state to an undiscounted cash flow."""

    def __init__(self) -> None:
        self.requires: frozenset[str] = frozenset()

    @abstractmethod
    def __call__(self, state: Mapping[str, Any]) -> float: ...


def _check_side(side: str) -> None:
    if side not in ("call", "put"):
        raise ValueError("side should be 'call' or 'put'")


class Vanilla(Payoff):
    """``max(x - k, 0)`` for a call or ``max(k - x, 0)`` for a put, with ``x = state[key]``."""

    def __init__(self, key: str, k: float, side: str) -> None:
        _check_side(side)
        super().__init__()
        self.key = key
        self.k = float(k)
        self.side = side
        self.requires = frozenset({self.key})

    def __call__(self, state: Mapping[str, Any]) -> float:
        x = float(state[self.key])
        if self.side == "call":
            return max(x - self.k, 0.0)
        return max(self.k - x, 0.0)


class Digital(Payoff):
    """Digital on ``x = state[key]``: pays if ``x > k`` (call) or ``x < k`` (put).

    The ``"cash"`` variant pays ``cash``; the ``"asset"`` variant pays ``x``.
    """

    def __init__(self, key: str, k: float, side: str, cash: float, variant: str) -> None:
        _check_side(side)
        if variant not in ("cash", "asset"):
            raise ValueError("variant should be 'cash' or 'asset'")
        super().__init__()
        self.key = key
        self.k = float(k)
        self.side = side
        self.variant = variant
        self.cash = float(cash)
        self.requires = frozenset({self.key})

    def __call__(self, state: Mapping[str, Any]) -> float:
        x = float(state[self.key])
        hit = x > self.k if self.side == "call" else x < self.k
        if not hit:
            return 0.0
        return self.cash if self.variant == "cash" else x


class AsianFloatingVanilla(Payoff):
    """Floating-strike payoff ``max(x - k, 0)`` with both ``x`` and ``k`` read from the state.

    With ``key1="terminal"`` and ``key2="average"`` this is the floating-strike
    Asian option, whose strike is the arithmetic average of the path.
    """

    def __init__(self, key1: str, key2: str, side: str) -> None:
        _check_side(side)
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.side = side
        self.requires = frozenset({key1, key2})

    def __call__(self, state: Mapping[str, Any]) -> float:
        x = float(state[self.key1])
        k = float(state[self.key2])
        if self.side == "call":
            return max(x - k, 0.0)
        return max(k - x, 0.0)


class KnockOutAnyN(Payoff):
    """Knock-out wrapper: pays ``inner`` unless any of the barrier flags in ``keys`` is set."""

    def __init__(self, inner: Payoff, keys: Iterable[str]) -> None:
        super().__init__()
        self.inner = inner
        self.keys = list(keys)
        self.requires = inner.requires | frozenset(self.keys)

    def __call__(self, state: Mapping[str, Any]) -> float:
        if any(state.get(key, False) for key in self.keys):
            return 0.0
        return float(self.inner(state))


class KnockInAllN(Payoff):
    """Knock-in wrapper: pays ``inner`` only if every barrier flag in ``keys`` is set."""

    def __init__(self, inner: Payoff, keys: Iterable[str]) -> None:
        super().__init__()
        self.inner = inner
        self.keys = list(keys)
        self.requires = inner.requires | frozenset(self.keys)

    def __call__(self, state: Mapping[str, Any]) -> float:
        if not all(state.get(key, False) for key in self.keys):
            return 0.0
        return float(self.inner(state))


class KnockInKnockOut(Payoff):
    """Knock-in/knock-out wrapper: pays ``inner`` iff ``ki_key`` was hit and ``ko_key`` was not."""

    def __init__(self, inner: Payoff, ki_key: str, ko_key: str) -> None:
        if ki_key == ko_key:
            raise ValueError("'ki_key' and 'ko_key' should be different")
        super().__init__()
        self.inner = inner
        self.ki_key = ki_key
        self.ko_key = ko_key
        self.requires = inner.requires | frozenset({ki_key, ko_key})

    def __call__(self, state: Mapping[str, Any]) -> float:
        if not state.get(self.ki_key, False) or state.get(self.ko_key, False):
            return 0.0
        return float(self.inner(state))
