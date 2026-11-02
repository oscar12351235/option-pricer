"""Monte Carlo prices checked against independent reference prices.

References:
- Heston / Bates European calls: QuantLib's semi-analytic AnalyticHestonEngine
  and BatesEngine.
- Merton jump-diffusion European call: the Merton (1976) series.
- American put under GBM: a Cox-Ross-Rubinstein binomial tree.

A Monte Carlo estimate passes when its error is within three standard errors
plus, where stated, an allowance for time discretisation or regression bias.
Every result is printed in the "benchmark results" section of the pytest
summary.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import QuantLib as ql

from option_pricer import (
    American,
    BatesModel,
    European,
    LognormalJumps,
    MonteCarloConfig,
    MonteCarloEngine,
    NoJump,
    NumbaHestonSampler,
    Vanilla,
)
from option_pricer.models import VectorizedJumpGBMSampler

Recorder = Callable[[str, float, float, float], None]

# Market and model parameters shared by the Heston and Bates benchmarks. The
# variance parameters violate the Feller condition (2 * kappa * theta = 0.12 <
# sigma^2 = 0.25), the regime in which the QE scheme matters most.
S0, STRIKE, MATURITY = 100.0, 100.0, 1.0
RATE, DIVIDEND = 0.03, 0.01
V0, KAPPA, THETA, VOL_OF_VAR, RHO = 0.04, 1.5, 0.04, 0.5, -0.7
JUMP_LAMBDA, JUMP_MU, JUMP_SIGMA = 0.4, -0.12, 0.18

# Weak (time-discretisation) error allowance for the QE scheme. With 4 million
# paths the measured bias at 25, 50 and 100 steps was below 0.1% of the price
# and statistically indistinguishable from zero.
QE_ALLOWANCE = 1e-3

HESTON_PATHS, HESTON_STEPS = 200_000, 100


@dataclass(frozen=True)
class QuantLibMarket:
    today: ql.Date
    maturity: ql.Date
    spot: ql.QuoteHandle
    rates: ql.YieldTermStructureHandle
    dividends: ql.YieldTermStructureHandle

    def call_price(self, engine: ql.PricingEngine) -> float:
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, STRIKE), ql.EuropeanExercise(self.maturity)
        )
        option.setPricingEngine(engine)
        return float(option.NPV())


@pytest.fixture(scope="module")
def market() -> QuantLibMarket:
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()  # today + 365 days is exactly T = 1
    return QuantLibMarket(
        today=today,
        maturity=today + 365,
        spot=ql.QuoteHandle(ql.SimpleQuote(S0)),
        rates=ql.YieldTermStructureHandle(ql.FlatForward(today, RATE, day_count, ql.Continuous)),
        dividends=ql.YieldTermStructureHandle(
            ql.FlatForward(today, DIVIDEND, day_count, ql.Continuous)
        ),
    )


def black_scholes(s0: float, k: float, t: float, r: float, q: float, vol: float, call: bool):
    d1 = (math.log(s0 / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    cdf = ql.CumulativeNormalDistribution()
    if call:
        return s0 * math.exp(-q * t) * cdf(d1) - k * math.exp(-r * t) * cdf(d2)
    return k * math.exp(-r * t) * cdf(-d2) - s0 * math.exp(-q * t) * cdf(-d1)


def merton_call(s0, k, t, r, q, vol, lam, mu, delta, n_terms=80):
    """Merton (1976) series: Poisson mixture of Black-Scholes prices."""
    kbar = math.exp(mu + 0.5 * delta * delta) - 1.0
    lam_prime = lam * (1.0 + kbar)
    total = 0.0
    for n in range(n_terms):
        weight = math.exp(-lam_prime * t + n * math.log(lam_prime * t) - math.lgamma(n + 1))
        r_n = r - lam * kbar + n * math.log(1.0 + kbar) / t
        vol_n = math.sqrt(vol * vol + n * delta * delta / t)
        total += weight * black_scholes(s0, k, t, r_n, q, vol_n, call=True)
    return total


def crr_put(s0, k, t, r, q, vol, n_steps, american):
    """Cox-Ross-Rubinstein binomial tree for a put, vectorised over nodes."""
    dt = t / n_steps
    u = math.exp(vol * math.sqrt(dt))
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)
    j = np.arange(n_steps + 1)
    values = np.maximum(k - s0 * u ** (2 * j - n_steps), 0.0)
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(i + 1)
            values = np.maximum(values, k - s0 * u ** (2 * j - i))
    return float(values[0])


def mc_price(sampler, payoff, exercise, n_paths, n_steps, seed, r=RATE, q=DIVIDEND, s0=S0):
    config = MonteCarloConfig(
        n_paths=n_paths,
        timegrid=np.linspace(0.0, MATURITY, n_steps + 1),
        seed=seed,
        antithetic=True,
    )
    model = BatesModel(sampler=sampler, S0=s0, r=r, q=q)
    engine = MonteCarloEngine(
        model=model, payoff=payoff, features=[], exercise=exercise, config=config
    )
    return engine.price()


def heston_sampler(jump=None) -> NumbaHestonSampler:
    return NumbaHestonSampler(
        S0=S0,
        v0=V0,
        kappa=KAPPA,
        theta=THETA,
        sigma=VOL_OF_VAR,
        rho=RHO,
        r=RATE,
        q=DIVIDEND,
        jump=jump,
    )


def assert_within_mc_error(estimate, stderr, reference, allowance=0.0):
    assert abs(estimate - reference) <= 3.0 * stderr + allowance, (
        f"MC {estimate:.5f} vs reference {reference:.5f}: error {estimate - reference:+.5f} "
        f"exceeds 3 SE ({3.0 * stderr:.5f}) + allowance ({allowance:.5f})"
    )


@pytest.mark.slow
def test_heston_call_matches_quantlib_analytic(market: QuantLibMarket, record_benchmark: Recorder):
    process = ql.HestonProcess(
        market.rates, market.dividends, market.spot, V0, KAPPA, THETA, VOL_OF_VAR, RHO
    )
    reference = market.call_price(ql.AnalyticHestonEngine(ql.HestonModel(process)))

    price, stderr = mc_price(
        heston_sampler(),
        Vanilla(key="terminal", k=STRIKE, side="call"),
        European(),
        HESTON_PATHS,
        HESTON_STEPS,
        seed=20260101,
    )
    record_benchmark("Heston European call (QE, Numba)", reference, price, stderr)

    assert stderr < 5e-3 * reference
    assert_within_mc_error(price, stderr, reference, allowance=QE_ALLOWANCE * reference)


@pytest.mark.slow
def test_bates_call_matches_quantlib_bates_engine(
    market: QuantLibMarket, record_benchmark: Recorder
):
    # LognormalJumps(lam, mu, sigma) maps to QuantLib's (lambda, nu, delta): in both
    # the log jump size is N(nu, delta^2) and the drift is compensated by
    # lambda * (exp(nu + delta^2 / 2) - 1). test_quantlib_bates_jump_convention_is_merton
    # checks this convention numerically.
    process = ql.BatesProcess(
        market.rates,
        market.dividends,
        market.spot,
        V0,
        KAPPA,
        THETA,
        VOL_OF_VAR,
        RHO,
        JUMP_LAMBDA,
        JUMP_MU,
        JUMP_SIGMA,
    )
    reference = market.call_price(ql.BatesEngine(ql.BatesModel(process)))

    price, stderr = mc_price(
        heston_sampler(LognormalJumps(lam=JUMP_LAMBDA, mu=JUMP_MU, sigma=JUMP_SIGMA)),
        Vanilla(key="terminal", k=STRIKE, side="call"),
        European(),
        HESTON_PATHS,
        HESTON_STEPS,
        seed=20260102,
    )
    record_benchmark("Bates European call (QE + jumps, Numba)", reference, price, stderr)

    assert stderr < 5e-3 * reference
    assert_within_mc_error(price, stderr, reference, allowance=QE_ALLOWANCE * reference)


def test_quantlib_bates_jump_convention_is_merton(market: QuantLibMarket):
    # With constant variance (v0 = theta, vol of variance ~ 0) Bates reduces to
    # Merton's jump-diffusion, whose series price uses log jumps N(mu, delta^2).
    vol = 0.2
    process = ql.BatesProcess(
        market.rates,
        market.dividends,
        market.spot,
        vol * vol,
        5.0,
        vol * vol,
        1e-4,
        0.0,
        JUMP_LAMBDA,
        JUMP_MU,
        JUMP_SIGMA,
    )
    quantlib_price = market.call_price(ql.BatesEngine(ql.BatesModel(process)))
    series = merton_call(
        S0, STRIKE, MATURITY, RATE, DIVIDEND, vol, JUMP_LAMBDA, JUMP_MU, JUMP_SIGMA
    )
    assert quantlib_price == pytest.approx(series, abs=1e-6)


def test_merton_call_matches_series(record_benchmark: Recorder):
    vol = 0.2
    reference = merton_call(
        S0, STRIKE, MATURITY, RATE, DIVIDEND, vol, JUMP_LAMBDA, JUMP_MU, JUMP_SIGMA
    )
    jump = LognormalJumps(lam=JUMP_LAMBDA, mu=JUMP_MU, sigma=JUMP_SIGMA)
    sampler = VectorizedJumpGBMSampler(S0=S0, r=RATE, q=DIVIDEND, sigma=vol, jump=jump)

    # GBM with jumps is simulated exactly on the grid: no discretisation allowance.
    price, stderr = mc_price(
        sampler, Vanilla(key="terminal", k=STRIKE, side="call"), European(), 200_000, 12, seed=7
    )
    record_benchmark("Merton jump-diffusion European call", reference, price, stderr)

    assert_within_mc_error(price, stderr, reference)


def test_crr_tree_converges_to_black_scholes():
    tree = crr_put(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, n_steps=4000, american=False)
    assert tree == pytest.approx(
        black_scholes(36.0, 40.0, 1.0, 0.06, 0.0, 0.2, call=False), abs=1e-3
    )


@pytest.mark.slow
def test_american_put_lsm_matches_crr_binomial(record_benchmark: Recorder):
    # Longstaff & Schwartz (2001), Table 1, first row: S0 = 36, K = 40, r = 6%,
    # sigma = 20%, T = 1, with 50 exercise dates per year.
    s0, k, r, q, vol = 36.0, 40.0, 0.06, 0.0, 0.2
    reference = crr_put(s0, k, MATURITY, r, q, vol, n_steps=4000, american=True)

    sampler = VectorizedJumpGBMSampler(S0=s0, r=r, q=q, sigma=vol, jump=NoJump())
    payoff = Vanilla(key="terminal", k=k, side="put")
    american, american_se = mc_price(
        sampler, payoff, American(), 100_000, 50, seed=11, r=r, q=q, s0=s0
    )
    european, european_se = mc_price(
        sampler, payoff, European(), 100_000, 50, seed=11, r=r, q=q, s0=s0
    )
    record_benchmark("American put, LSM vs CRR (4000 steps)", reference, american, american_se)

    # LSM is low-biased (suboptimal regression-based exercise on a discrete grid
    # of 50 dates), so only the lower bound gets the extra 1% allowance.
    assert reference - 3.0 * american_se - 0.01 * reference <= american
    assert american <= reference + 3.0 * american_se
    assert american >= european
    assert american >= black_scholes(s0, k, MATURITY, r, q, vol, call=False)


@pytest.mark.parametrize(
    "label, jump",
    [
        ("Heston", None),
        ("Bates", LognormalJumps(lam=JUMP_LAMBDA, mu=JUMP_MU, sigma=JUMP_SIGMA)),
    ],
)
def test_discounted_spot_is_martingale(label, jump, record_benchmark: Recorder):
    # A call with zero strike pays S_T, so its price is E[exp(-rT) S_T] = S0 exp(-qT).
    reference = S0 * math.exp(-DIVIDEND * MATURITY)
    price, stderr = mc_price(
        heston_sampler(jump),
        Vanilla(key="terminal", k=0.0, side="call"),
        European(),
        100_000,
        50,
        seed=3,
    )
    record_benchmark(f"{label} martingale E[exp(-rT) S_T]", reference, price, stderr)

    assert_within_mc_error(price, stderr, reference)
