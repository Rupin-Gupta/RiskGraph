"""Trade schema (SPEC §3.1)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat, model_validator

Desk = Literal["fx", "rates", "equity_derivatives"]
InstrumentType = Literal[
    "fx_forward", "fx_spot", "interest_rate_swap", "treasury_bond", "european_option", "cash_equity"
]

DESK: dict[str, Desk] = {
    "fx_forward": "fx",
    "fx_spot": "fx",
    "interest_rate_swap": "rates",
    "treasury_bond": "rates",
    "european_option": "equity_derivatives",
    "cash_equity": "equity_derivatives",
}
# Fields each instrument type must set. Every other field below must be None.
REQUIRED: dict[str, set[str]] = {
    "fx_forward": {"currency_pair", "forward_rate", "maturity_date", "counterparty_id"},
    "fx_spot": {"currency_pair", "forward_rate", "maturity_date"},
    "interest_rate_swap": {"fixed_rate", "pay_receive", "maturity_date", "counterparty_id"},
    "treasury_bond": {"coupon", "maturity_date"},
    "european_option": {"underlying", "strike", "option_type", "maturity_date"},
    "cash_equity": {"underlying"},
}
INSTRUMENT_FIELDS = set().union(*REQUIRED.values())


class Trade(BaseModel):
    """One trade.

    Units: `notional` is signed (+ long / buy base, - short / sell base). FX: units of the
    pair's base currency (EUR for EURUSD, USD for USDINR). Swaps: USD, positive, direction
    from `pay_receive` (payer pays fixed). Bonds: USD face. Equities and options: USD notional
    of the underlying (units held = notional / spot). `fixed_rate` and `coupon`: decimal per
    year, paid semi-annually. `strike`: underlying price. `forward_rate`: contract rate in
    quote units per base unit (USD per EUR, INR per USD); for spot trades, the dealt rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str
    desk: Desk
    instrument_type: InstrumentType
    counterparty_id: str | None = None
    currency: str
    notional: float
    trade_date: date
    maturity_date: date | None = None
    strike: PositiveFloat | None = None
    option_type: Literal["call", "put"] | None = None
    underlying: Literal["SPY", "AAPL", "MSFT", "JPM"] | None = None
    fixed_rate: float | None = None
    pay_receive: Literal["payer", "receiver"] | None = None
    coupon: float | None = None
    currency_pair: Literal["EURUSD", "USDINR"] | None = None
    forward_rate: PositiveFloat | None = None

    @model_validator(mode="after")
    def _fields_match_instrument(self) -> Trade:
        need = REQUIRED[self.instrument_type]
        missing = sorted(f for f in need if getattr(self, f) is None)
        extra = sorted(f for f in INSTRUMENT_FIELDS - need if getattr(self, f) is not None)
        if missing or extra:
            raise ValueError(f"{self.instrument_type}: missing {missing}, unexpected {extra}")
        if self.desk != DESK[self.instrument_type]:
            raise ValueError(f"{self.instrument_type} belongs to desk {DESK[self.instrument_type]}")
        if self.maturity_date is not None and self.maturity_date <= self.trade_date:
            raise ValueError("maturity_date must be after trade_date")
        if self.instrument_type == "interest_rate_swap" and self.notional <= 0:
            raise ValueError("swap notional must be positive; use pay_receive for direction")
        return self
