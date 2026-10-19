from __future__ import annotations

from abc import ABC, abstractmethod


class Payoff(ABC):
    def __init__(self):
        self.requires: frozenset[str] = frozenset()

    @abstractmethod
    def __call__(self, state: dict) -> float: ...


class Vanilla(Payoff):
    def __init__(self, key: str, k: float, side: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        super().__init__()
        self.key = key
        self.k = float(k)
        self.side = side
        self.requires = frozenset({self.key})

    def __call__(self, state):
        x = float(state[self.key])
        if self.side == "call":
            return max(x - self.k, 0.0)
        return max(self.k - x, 0.0)


class Digital(Payoff):
    def __init__(self, key: str, k: float, side: str, cash: float, variant: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        if variant not in ("cash", "asset"):
            raise ValueError("variant should be 'cash' or 'asset'")
        super().__init__()
        self.key = key
        self.k = float(k)
        self.side = side
        self.variant = variant
        self.cash = float(cash)
        self.requires = frozenset({self.key})

    def __call__(self, state):
        x = float(state[self.key])
        if self.side == "call":
            if x > self.k:
                return self.cash if self.variant == "cash" else x
            return 0.0
        # put
        if x < self.k:
            return self.cash if self.variant == "cash" else x
        return 0.0


class AsianFloatingVanilla(Payoff):
    def __init__(self, key1: str, key2: str, side: str):
        if side not in ("call", "put"):
            raise ValueError("side should be 'call' or 'put'")
        super().__init__()
        self.key1 = key1
        self.key2 = key2
        self.side = side
        self.requires = frozenset({key1, key2})

    def __call__(self, state):
        x = float(state[self.key1])
        k = float(state[self.key2])
        if self.side == "call":
            return max(x - k, 0.0)
        return max(k - x, 0.0)


class KnockOutAnyN(Payoff):
    def __init__(self, inner: Payoff, keys):
        super().__init__()
        self.inner = inner
        self.keys = list(keys)
        unique_keys = list(dict.fromkeys(self.keys))
        self.requires = inner.requires | frozenset(unique_keys)

    def __call__(self, state):
        for k in self.keys:
            if state.get(k, False):
                return 0.0
        return float(self.inner(state))


class KnockInAllN(Payoff):
    def __init__(self, inner: Payoff, keys):
        super().__init__()
        self.inner = inner
        self.keys = list(keys)
        unique_keys = list(dict.fromkeys(self.keys))
        self.requires = inner.requires | frozenset(unique_keys)

    def __call__(self, state):
        for k in self.keys:
            if not state.get(k, False):
                return 0.0
        return float(self.inner(state))


class KnockInKnockOut(Payoff):
    def __init__(self, inner: Payoff, ki_key, ko_key):
        if ki_key == ko_key:
            raise ValueError("'ki_key' and 'ko_key' should be different")
        super().__init__()
        self.inner = inner
        self.ki_key = ki_key
        self.ko_key = ko_key
        self.requires = inner.requires | frozenset({ki_key, ko_key})

    def __call__(self, state):
        if not state.get(self.ki_key, False):
            return 0.0
        if state.get(self.ko_key, False):
            return 0.0
        return float(self.inner(state))
