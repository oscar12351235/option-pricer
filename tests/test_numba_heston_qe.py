from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np
import pytest

from option_pricer import (
    BatesModel,
    European,
    LognormalJumps,
    MonteCarloConfig,
    MonteCarloEngine,
    NoJump,
    NumbaHestonSampler,
    Vanilla,
)
from option_pricer.models.numba_heston_qe import (
    _poisson_from_uniform,
    _std_normal_cdf,
    qe_variance_step,
    simulate_bates_qe,
)

HESTON: dict[str, Any] = dict(
    S0=100.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7, r=0.03, q=0.01
)
JUMPS = LognormalJumps(lam=0.4, mu=-0.12, sigma=0.18)
GRID = np.linspace(0.0, 1.0, 21)


def cir_conditional_moments(v, dt, kappa, theta, sigma):
    e = math.exp(-kappa * dt)
    mean = theta + (v - theta) * e
    var = v * sigma**2 * e / kappa * (1 - e) + theta * sigma**2 / (2 * kappa) * (1 - e) ** 2
    return mean, var


@pytest.mark.parametrize(
    "v, dt, kappa, theta, sigma, quadratic",
    [
        (0.04, 0.01, 1.5, 0.04, 0.5, True),  # psi << 1.5
        (0.0005, 0.25, 0.5, 0.04, 1.0, False),  # psi > 1.5: near-zero variance, large dt
    ],
)
def test_qe_step_matches_cir_conditional_mean_and_variance(v, dt, kappa, theta, sigma, quadratic):
    *_, branch = qe_variance_step.py_func(v, dt, kappa, theta, sigma, 0.3)
    assert branch is quadratic

    n = 400_000
    z = np.random.default_rng(5).standard_normal((1, n, 2))
    sampler = NumbaHestonSampler(
        S0=100.0, v0=v, kappa=kappa, theta=theta, sigma=sigma, rho=0.0, r=0.0, q=0.0
    )
    _, variances = sampler.simulate(n, np.array([0.0, dt]), np.random.default_rng(0), z=z)
    v_next = variances[:, 1]

    mean, var = cir_conditional_moments(v, dt, kappa, theta, sigma)
    centred_sq = (v_next - v_next.mean()) ** 2
    assert abs(v_next.mean() - mean) <= 5 * v_next.std() / math.sqrt(n)
    assert abs(centred_sq.mean() - var) <= 5 * centred_sq.std() / math.sqrt(n)
    assert np.all(v_next >= 0.0)


def test_exponential_branch_puts_mass_at_zero():
    v_next, _, p, beta, quadratic = qe_variance_step.py_func(0.0005, 0.25, 0.5, 0.04, 1.0, -3.0)
    assert not quadratic
    assert 0.0 < p < 1.0 and beta > 0.0
    assert _std_normal_cdf.py_func(-3.0) <= p
    assert v_next == 0.0


def test_std_normal_cdf_values():
    assert _std_normal_cdf.py_func(0.0) == pytest.approx(0.5)
    assert _std_normal_cdf.py_func(1.959963984540054) == pytest.approx(0.975)


def test_poisson_inversion_recovers_mean_and_terminates_at_u_equal_one():
    mean = 0.3
    u = (np.arange(200_000) + 0.5) / 200_000  # stratified uniforms
    counts = np.array([_poisson_from_uniform.py_func(x, mean) for x in u[::50]])
    assert counts.mean() == pytest.approx(mean, abs=0.01)
    assert _poisson_from_uniform.py_func(math.exp(-mean) * 0.999, mean) == 0
    assert _poisson_from_uniform.py_func(math.exp(-mean) * 1.001, mean) == 1
    assert _poisson_from_uniform.py_func(1.0, mean) > 0  # loop stops when the pmf underflows


@pytest.mark.parametrize("jump", [None, JUMPS])
def test_compiled_kernel_matches_python_reference(jump):
    sampler = NumbaHestonSampler(**HESTON, jump=jump)
    z = np.random.default_rng(1).standard_normal((len(GRID) - 1, 50, sampler.n_factors))
    args = (
        sampler.S0,
        sampler.v0,
        sampler.kappa,
        sampler.theta,
        sampler.sigma,
        sampler.rho,
        sampler.r,
        sampler.q,
        sampler.jump_lambda,
        sampler.jump_mu,
        sampler.jump_sigma,
        GRID,
        z,
    )
    compiled = simulate_bates_qe(*args, True)
    reference = simulate_bates_qe.py_func(*args, True)
    np.testing.assert_allclose(compiled[0], reference[0], rtol=1e-12)
    np.testing.assert_allclose(compiled[1], reference[1], rtol=1e-12, atol=1e-15)

    spots_only, empty = simulate_bates_qe.py_func(*args, False)
    np.testing.assert_allclose(spots_only, reference[0], rtol=1e-12)
    assert empty.shape == (0, 0)


@pytest.mark.parametrize("jump", [None, NoJump(), JUMPS])
def test_paths_have_expected_shape_and_initial_values(jump):
    sampler = NumbaHestonSampler(**HESTON, jump=jump)
    spots, variances = sampler.simulate(1_000, GRID, np.random.default_rng(3))

    assert spots.shape == variances.shape == (1_000, len(GRID))
    assert np.all(spots[:, 0] == HESTON["S0"])
    assert np.all(variances[:, 0] == HESTON["v0"])
    assert np.all(spots > 0.0) and np.all(np.isfinite(spots))
    assert np.all(variances >= 0.0)
    assert sampler.n_factors == (4 if jump is JUMPS else 2)


def test_variance_stays_non_negative_when_feller_condition_fails():
    # 2 * kappa * theta = 0.02 << sigma^2 = 1: the variance hits zero often.
    sampler = NumbaHestonSampler(
        S0=100.0, v0=0.01, kappa=0.5, theta=0.02, sigma=1.0, rho=-0.9, r=0.0, q=0.0
    )
    _, variances = sampler.simulate(20_000, np.linspace(0.0, 2.0, 25), np.random.default_rng(4))
    assert variances.min() == 0.0
    assert np.mean(variances[:, 1:] == 0.0) > 0.01


def test_sampler_is_reproducible_and_seed_dependent():
    sampler = NumbaHestonSampler(**HESTON, jump=JUMPS)
    a = sampler(500, GRID, np.random.default_rng(42))
    b = sampler(500, GRID, np.random.default_rng(42))
    c = sampler(500, GRID, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_output_does_not_depend_on_number_of_threads():
    sampler = NumbaHestonSampler(**HESTON, jump=JUMPS)
    reference = sampler(2_000, GRID, np.random.default_rng(8))
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single_thread = sampler(2_000, GRID, np.random.default_rng(8))
    finally:
        numba.set_num_threads(previous)
    np.testing.assert_array_equal(reference, single_thread)


def test_explicit_normals_are_used_and_validated():
    sampler = NumbaHestonSampler(**HESTON)
    z = np.random.default_rng(9).standard_normal((len(GRID) - 1, 10, 2))
    np.testing.assert_array_equal(
        sampler(10, GRID, np.random.default_rng(0), z=z),
        sampler(10, GRID, np.random.default_rng(1), z=z),
    )
    assert not np.allclose(
        sampler(10, GRID, np.random.default_rng(0), z=z),
        sampler(10, GRID, np.random.default_rng(0), z=-z),
    )
    with pytest.raises(ValueError, match="z must have shape"):
        sampler(10, GRID, np.random.default_rng(0), z=z[:, :, :1])


@pytest.mark.parametrize(
    "override, message",
    [
        ({"S0": 0.0}, "S0 must be positive"),
        ({"v0": -0.01}, "v0 must be non-negative"),
        ({"kappa": 0.0}, "kappa, theta and sigma"),
        ({"theta": -0.1}, "kappa, theta and sigma"),
        ({"sigma": 0.0}, "kappa, theta and sigma"),
        ({"rho": 1.5}, "rho must lie"),
        ({"jump": LognormalJumps(lam=-1.0, mu=0.0, sigma=0.1)}, "non-negative"),
    ],
)
def test_invalid_parameters_are_rejected(override, message):
    with pytest.raises(ValueError, match=message):
        NumbaHestonSampler(**{**HESTON, **override})


def test_unsupported_jump_type_is_rejected():
    class OtherJump:
        def compensator(self) -> float:
            return 0.0

        def step_multiplier(self, n, dt, randnumbergenerator, z=None):
            return np.ones(n)

    with pytest.raises(TypeError, match="jump must be"):
        NumbaHestonSampler(**HESTON, jump=OtherJump())


def _engine(antithetic: bool, use_crn: bool, jump=None, seed: int = 21) -> MonteCarloEngine:
    sampler = NumbaHestonSampler(**HESTON, jump=jump)
    return MonteCarloEngine(
        model=BatesModel(sampler=sampler, S0=100.0, r=0.03, q=0.01),
        payoff=Vanilla(key="terminal", k=100.0, side="call"),
        features=[],
        exercise=European(),
        config=MonteCarloConfig(
            n_paths=20_000, timegrid=GRID, seed=seed, use_crn=use_crn, antithetic=antithetic
        ),
    )


def test_engine_passes_correctly_shaped_normals_for_bates():
    engine = _engine(antithetic=True, use_crn=True, jump=JUMPS)
    assert engine._z_crn is not None
    assert engine._z_crn.shape == (len(GRID) - 1, 20_000, 4)
    np.testing.assert_array_equal(engine._z_crn[:, :10_000], -engine._z_crn[:, 10_000:])


def test_common_random_numbers_reuse_the_same_paths():
    with_crn = _engine(antithetic=False, use_crn=True)
    np.testing.assert_array_equal(with_crn._simulate_paths(), with_crn._simulate_paths())

    without_crn = _engine(antithetic=False, use_crn=False)
    assert not np.allclose(without_crn._simulate_paths(), without_crn._simulate_paths())


def test_antithetic_variates_reduce_the_standard_error():
    price_plain, se_plain = _engine(antithetic=False, use_crn=False).price()
    price_anti, se_anti = _engine(antithetic=True, use_crn=False).price()
    assert se_anti < 0.8 * se_plain
    assert abs(price_anti - price_plain) < 4 * math.hypot(se_plain, se_anti)


def test_martingale_correction_falls_back_when_mgf_does_not_exist():
    # One 20-year step with rho = 1: in the exponential branch A = K2 + K4 / 2 = 6
    # exceeds beta (about 3.7), so E[exp(A V_{t+dt})] is infinite and the
    # uncorrected drift K0 is used instead.
    v, dt, kappa, theta, sigma, rho = 0.0, 20.0, 1.0, 0.04, 1.0, 1.0
    _, _, _, beta, quadratic = qe_variance_step.py_func(v, dt, kappa, theta, sigma, 0.0)
    big_a = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    assert not quadratic and big_a >= beta

    z = np.random.default_rng(2).standard_normal((1, 8, 2))
    spots, variances = simulate_bates_qe.py_func(
        100.0, v, kappa, theta, sigma, rho, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0, dt]), z, True
    )
    assert np.all(np.isfinite(spots)) and np.all(spots > 0.0)
    assert np.all(variances >= 0.0)
