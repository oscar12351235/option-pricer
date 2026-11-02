from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    reference: float
    estimate: float
    stderr: float

    @property
    def error(self) -> float:
        return self.estimate - self.reference


_RESULTS = pytest.StashKey[list[BenchmarkResult]]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_RESULTS] = []


@pytest.fixture
def record_benchmark(request: pytest.FixtureRequest) -> Callable[[str, float, float, float], None]:
    """Record a Monte Carlo estimate next to its reference value for the summary table."""
    results = request.config.stash[_RESULTS]

    def record(name: str, reference: float, estimate: float, stderr: float) -> None:
        results.append(BenchmarkResult(name, reference, estimate, stderr))

    return record


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    results = config.stash.get(_RESULTS, [])
    if not results:
        return
    terminalreporter.section("benchmark results")
    terminalreporter.write_line(
        f"{'benchmark':<42} {'reference':>10} {'MC':>10} {'SE':>8} {'error':>9} {'|err|/SE':>8}"
    )
    for res in results:
        ratio = abs(res.error) / res.stderr if res.stderr > 0 else float("nan")
        terminalreporter.write_line(
            f"{res.name:<42} {res.reference:>10.4f} {res.estimate:>10.4f} {res.stderr:>8.4f} "
            f"{res.error:>+9.4f} {ratio:>8.2f}"
        )
