import numpy as np
import pytest

from riskgraph.pricing.curve import discount_factors, zero_rates
from riskgraph.pricing.fx import forward_pv, forward_rate
from riskgraph.pricing.rates import (
    bond_cashflows,
    coupon_times,
    dv01,
    par_swap_rate,
    pv,
    swap_cashflows,
)

FLAT = np.array([[0.04, 0.04, 0.04]])
CURVE = np.array([[0.045, 0.040, 0.042]])  # 2Y, 5Y, 10Y


def test_curve_is_linear_between_pillars_and_flat_outside() -> None:
    z = zero_rates(CURVE, [1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 30.0])
    np.testing.assert_allclose(z[0], [0.045, 0.045, 0.0425, 0.040, 0.041, 0.042, 0.042])
    # Semi-annual compounding: DF(t) = (1 + z/2)^(-2t).
    assert discount_factors(FLAT, [3.0])[0, 0] == pytest.approx(1.02**-6)


def test_coupon_times_count_back_from_maturity() -> None:
    np.testing.assert_allclose(coupon_times(2.0), [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(coupon_times(1.25), [0.25, 0.75, 1.25])
    assert len(coupon_times(731 / 365.25)) == 4  # a ~1-day stub merges into the next period


@pytest.mark.parametrize("t", [2.0, 5.0, 10.0])
def test_bond_at_par_when_coupon_equals_yield(t: float) -> None:
    assert pv(*bond_cashflows(100.0, 0.04, t), FLAT)[0] == pytest.approx(100.0, abs=1e-10)


@pytest.mark.parametrize("t", [2.0, 5.0, 10.0])
def test_swap_pv_zero_at_par_rate(t: float) -> None:
    # On a flat curve the par rate equals the curve rate (same semi-annual compounding).
    assert par_swap_rate(t, FLAT)[0] == pytest.approx(0.04, abs=1e-12)
    k = float(par_swap_rate(t, CURVE)[0])
    for payer in (True, False):
        assert pv(*swap_cashflows(1e8, k, t, payer), CURVE)[0] == pytest.approx(0.0, abs=1e-6)


def test_swap_dv01_sign_and_symmetry() -> None:
    payer = dv01(*swap_cashflows(1e8, 0.04, 10.0, True), CURVE)[0]
    receiver = dv01(*swap_cashflows(1e8, 0.04, 10.0, False), CURVE)[0]
    assert payer > 0 and receiver == pytest.approx(-payer)


def test_bond_dv01_matches_modified_duration() -> None:
    # Par 10Y bond at a flat 4%: modified duration = (1 - 1.02^-20) / 0.04 = 8.1757 years,
    # so +1bp moves 100 face by about -8.1757 * 100 * 1e-4 (convexity adds ~0.02%).
    expected = -(1 - 1.02**-20) / 0.04 * 100 * 1e-4
    assert dv01(*bond_cashflows(100.0, 0.04, 10.0), FLAT)[0] == pytest.approx(expected, rel=1e-3)


def test_fx_forward_known_value_and_zero_at_fair_forward() -> None:
    # EURUSD spot 1.10, 1Y, USD 4% and EUR 2% (semi-annual): F = 1.10 * 1.02^2 / 1.01^2.
    df_usd, df_eur = 1.02**-2, 1.01**-2
    f = forward_rate(1.10, df_eur, df_usd)
    assert float(f) == pytest.approx(1.10 * 1.02**2 / 1.01**2)
    assert float(forward_pv(1e6, float(f), 1.10, df_eur, df_usd)) == pytest.approx(0.0, abs=1e-8)
    # Long 1M EUR at 1.10 is worth 1M * (F - 1.10), paid in one year, in USD today.
    expected = 1e6 * (1.10 * 1.02**2 / 1.01**2 - 1.10) * 1.02**-2
    assert float(forward_pv(1e6, 1.10, 1.10, df_eur, df_usd)) == pytest.approx(expected)
