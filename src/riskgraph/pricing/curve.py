"""USD Treasury curve from the DGS2/DGS5/DGS10 pillars (SPEC §4.1).

ponytail: CMT par yields are used directly as zero rates (no bootstrap). Differences are a
few bp; a bootstrapped or OIS curve is the upgrade path (Appendix B, single-curve USD).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PILLARS = np.array([2.0, 5.0, 10.0])  # years


def zero_rates(pillar_rates: FloatArray, t: npt.ArrayLike) -> FloatArray:
    """Zero rates (decimal, semi-annual compounding), shape (n_scenarios, len(t)).

    pillar_rates: shape (n_scenarios, 3), decimal yields at 2Y/5Y/10Y. t: years. Linear
    interpolation between pillars, flat extrapolation outside them.
    """
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    # Interpolation is linear in the pillar values, so its weights depend on t only.
    weights = np.stack([np.interp(tt, PILLARS, e) for e in np.eye(len(PILLARS))])
    return pillar_rates @ weights


def discount_factors(pillar_rates: FloatArray, t: npt.ArrayLike) -> FloatArray:
    """Discount factors (1 + z/2)^(-2t), shape (n_scenarios, len(t)). t in years."""
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return (1 + zero_rates(pillar_rates, tt) / 2) ** (-2 * tt)
