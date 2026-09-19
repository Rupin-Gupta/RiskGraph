import numpy as np
import pytest

from riskgraph.pricing.black_scholes import bs_greeks, bs_price


def test_hull_example_15_6() -> None:
    # Hull, Options, Futures, and Other Derivatives, Example 15.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
    assert float(bs_price(42.0, 40.0, 0.5, 0.10, 0.20, True)) == pytest.approx(4.76, abs=5e-3)
    assert float(bs_price(42.0, 40.0, 0.5, 0.10, 0.20, False)) == pytest.approx(0.81, abs=5e-3)


def test_hull_greeks_example() -> None:
    # Hull, "The Greek Letters" running example: S=49, K=50, r=5%, sigma=20%, T=20 weeks.
    # Price 2.40, delta 0.522, gamma 0.066, vega 12.1 (per 1.00 vol), theta -4.31 per year.
    args = (49.0, 50.0, 20 / 52, 0.05, 0.20, True)
    g = bs_greeks(*args)
    assert float(bs_price(*args)) == pytest.approx(2.40, abs=5e-3)
    assert float(g["delta"]) == pytest.approx(0.522, abs=5e-4)
    assert float(g["gamma"]) == pytest.approx(0.066, abs=5e-4)
    assert float(g["vega"]) == pytest.approx(12.1, abs=0.05)
    assert float(g["theta"]) == pytest.approx(-4.31, abs=5e-3)


def test_atm_textbook_values() -> None:
    # S=K=100, T=1, r=5%, sigma=20%: call 10.4506, put 5.5735 (widely tabulated).
    assert float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, True)) == pytest.approx(10.4506, abs=1e-4)
    assert float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, False)) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity() -> None:
    rng = np.random.default_rng(0)
    s, k = rng.uniform(50, 150, 20), rng.uniform(50, 150, 20)
    t, r, v = rng.uniform(0.05, 2, 20), rng.uniform(0, 0.08, 20), rng.uniform(0.1, 0.6, 20)
    c, p = bs_price(s, k, t, r, v, True), bs_price(s, k, t, r, v, False)
    np.testing.assert_allclose(c - p, s - k * np.exp(-r * t), atol=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize(
    ("s", "k", "t", "r", "v"),
    [(100, 100, 1.0, 0.05, 0.2), (49, 50, 0.3846, 0.05, 0.2), (420, 380, 0.08, 0.04, 0.35)],
)
def test_greeks_match_finite_differences(
    s: float, k: float, t: float, r: float, v: float, is_call: bool
) -> None:
    def f(s: float = s, t: float = t, v: float = v) -> float:
        return float(bs_price(s, k, t, r, v, is_call))

    hs, ht, hv = 1e-3 * s, 1e-5, 1e-5
    fd = {
        "delta": (f(s=s + hs) - f(s=s - hs)) / (2 * hs),
        "gamma": (f(s=s + hs) - 2 * f() + f(s=s - hs)) / hs**2,
        "vega": (f(v=v + hv) - f(v=v - hv)) / (2 * hv),
        "theta": -(f(t=t + ht) - f(t=t - ht)) / (2 * ht),  # calendar time runs against T
    }
    g = bs_greeks(s, k, t, r, v, is_call)
    for name, value in fd.items():
        assert float(g[name]) == pytest.approx(value, rel=1e-4), name
