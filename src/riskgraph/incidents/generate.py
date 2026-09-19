"""Incident injection (SPEC §11.1): labeled cases on 2022-2025 dates, each validated by running
the daily risk process with its overrides.

Four types in equal numbers (ADR-010):
- position_jump: one new trade (book override) sized so its desk's VaR breaches the limit.
- market_shock: a real breach day; book and market data untouched.
- bad_data: one run-date print corrupted (spike or sign flip) below the hard-rule thresholds, so
  only the anomaly model flags it (a warning: the factor stays live) and it tips a limit into
  breach. Stale prints give zero shocks and cannot raise VaR, so they are not used.
- control: no injection; 40% near misses (90-99% utilization), the rest quiet days.

Every case sits on its own date. Incident IDs are assigned in random order, so an ID says
nothing about the type. The agents see only incident.json; labels live in ground_truth.json.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from riskgraph import cli
from riskgraph.book.generate import generate as generate_book
from riskgraph.book.generate import load_book
from riskgraph.book.schema import Trade
from riskgraph.marketdata.controls import fit_forest, run_controls
from riskgraph.pricing.market import CURVE, FACTORS
from riskgraph.risk.daily import hold, hs_var
from riskgraph.risk.factors import RiskContext
from riskgraph.risk.var import SCOPES

OUT = Path("data/synthetic/incidents")
HISTORY = Path("data/runs/backtest/history.parquet")
YEARS = ("2022-01-01", "2025-12-31")
DEV_SHARE = 0.3
NEAR_MISS_SHARE = 0.4
LABELS = {  # type -> (root cause, recommended action), SPEC §11.1
    "position_jump": ("position_change", "escalate_to_risk_manager"),
    "market_shock": ("market_move", "escalate_to_risk_manager"),
    "bad_data": ("bad_market_data", "route_to_data_ops"),
    "control": ("no_true_breach", "no_action"),
}
KINDS = {
    "fx": ("fx_forward", "fx_spot"),
    "rates": ("interest_rate_swap", "treasury_bond"),
    "equity_derivatives": ("european_option", "cash_equity"),
}
PREFIX = {"fx_forward": "FXF", "fx_spot": "FXS", "interest_rate_swap": "IRS"}
PREFIX |= {"treasury_bond": "BND", "european_option": "OPT", "cash_equity": "EQ"}
DESK_OF = {f: "fx" for f in ("EURUSD=X", "INR=X")} | {f: "rates" for f in CURVE}
DESK_OF |= {f: "equity_derivatives" for f in ("SPY", "AAPL", "MSFT", "JPM", "VIXCLS")}
FILL_FACTOR = "EURUSD=X"
FX_GAP = 0.019  # max corrupted gap to the second source (tolerance 200bp)
TRIES_PER_FACTOR = 150
MAX_TRIES = 2000


def expected_sections(kind: str, alert: Mapping[str, Any], near_miss: bool = False) -> list[str]:
    """Policy sections a correct escalation note should cite (corpus/synthetic, SPEC §9.1)."""
    scope_rule = (
        "MRLP-6.3"
        if alert["metric"] == "stress_loss"
        else ("MRLP-6.2" if alert["scope"] == "firm" else "MRLP-6.1")
    )
    return {
        "position_jump": ["MRLP-5.1", "MRLP-6.4", scope_rule],
        "market_shock": ["MRLP-5.2", "MRLP-6.5", scope_rule],
        "bad_data": ["MRLP-5.3", "MRLP-6.6", "MDCP-5.1"],
        "control": ["MRLP-5.4", "MRLP-4.3" if near_miss else "MRLP-4.4"],
    }[kind]


def base_utilization(limits: Mapping[str, Any]) -> pd.DataFrame:
    """Utilization of every limit per date from the backtest history (`make backtest`)."""
    if not HISTORY.exists():
        raise FileNotFoundError(f"{HISTORY} missing: run `make backtest` first")
    h = pd.read_parquet(HISTORY).set_index("asof").loc[YEARS[0] : YEARS[1]]
    var = limits["var_99_1d"]
    cols = {f"{s}/var_99_1d": h[f"var_hs_{s}"] / lim for s, lim in var["desks"].items()}
    cols["firm/var_99_1d"] = h["var_hs_firm"] / var["firm"]
    cols["firm/stress_loss"] = h["stress_firm"] / limits["stress_loss"]["firm"]
    return pd.DataFrame(cols)


def alert_of(limits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The limit row an alert is raised on: the highest utilization."""
    r = max(limits, key=lambda x: x["utilization"])
    return {k: r[k] for k in ("scope", "metric", "value", "limit", "utilization", "status")}


def top_factors(doc: Mapping[str, Any], scope: str, k: int = 3) -> list[str]:
    rows = doc["explain"][scope]["top_factors"]
    return [r["factor"] for r in rows if r["factor"] in FACTORS][:k]


def run_case(day: str, book: Path | None, market: Path | None, seed: int) -> dict[str, Any]:
    """Full daily run with the case's overrides (no database writes), quietly."""
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.daily_run(day, book, market, seed, db=False)


def check(kind: str, doc: Mapping[str, Any], base_u: float, extra: Mapping[str, Any]) -> str | None:
    """Why the validated run does not match the case type, or None if it does."""
    alert = alert_of(doc["limits"])
    findings = doc["controls"]["findings"]
    if doc["controls"]["excluded_factors"]:
        return "critical data-quality finding on the run date"
    top = top_factors(doc, alert["scope"])
    hit = {f["factor"] for f in findings} & set(top)
    if hit and kind != "bad_data":  # the policy would route it to Data Operations (MDCP-5.1)
        return f"finding on contributing factor {sorted(hit)}"
    if kind == "control":
        want = "warning" if extra["near_miss"] else "ok"
        return None if alert["status"] == want else f"status {alert['status']} != {want}"
    if alert["status"] != "breach":
        return f"alert status {alert['status']}"
    if kind == "market_shock":
        return None
    if base_u >= 1:
        return "limit already breached before the injection"
    if kind == "position_jump":
        pos = doc["explain"][alert["scope"]]["position"]
        return None if pos > 0 else f"position effect {pos:.0f} <= 0"
    f = extra["factor"]
    if not any(x["factor"] == f and x["severity"] == "warning" for x in findings):
        return f"no warning finding on {f}"
    return None if f in top else f"{f} not in top factors {top}"


class Generator:
    """Seeded search over dates, trades, and corruptions; every accepted case is validated."""

    def __init__(self, seed: int, out: Path) -> None:
        self.seed, self.out = seed, out
        self.rng = np.random.default_rng(seed)
        self.cfgs = cli.configs()
        self.panel = cli.market_panel()
        self.ctx = RiskContext.from_panel(self.panel, self.cfgs.risk)
        self.book = load_book(cli.BOOK)
        cc = self.cfgs.risk["controls"]
        self.forest = fit_forest(self.panel, cc, seed)
        crit = run_controls(self.panel, cc, self.forest)
        crit = crit[crit["severity"] == "critical"]["date"]
        u = base_utilization(self.cfgs.limits)
        self.util = u[~u.index.isin(pd.DatetimeIndex(crit)) & u.index.isin(self.ctx.levels.index)]
        self.used: set[pd.Timestamp] = set()
        self.cases: list[dict[str, Any]] = []
        self.tmp = out / "_tmp"

    # -- helpers ---------------------------------------------------------------------------
    def pool(self, mask: pd.Series) -> list[pd.Timestamp]:
        dates = [d for d in self.util.index[mask.to_numpy()] if d not in self.used]
        return [dates[i] for i in self.rng.permutation(len(dates))]

    def var_util(self, ctx: RiskContext, trades: Sequence[Trade], day: pd.Timestamp) -> dict:
        """Fast HS VaR utilization per scope (search only; validation runs the full process)."""
        hs = hs_var(hold(ctx, trades, ctx.ref_spots(trades), day), 0.99, 0.975)
        var = self.cfgs.limits["var_99_1d"]
        lim = var["desks"] | {"firm": var["firm"]}
        return {s: hs[j][0] / lim[s] for j, s in enumerate(SCOPES)}

    def accept(
        self,
        kind: str,
        day: pd.Timestamp,
        files: Mapping[str, Callable[[Path], None]],
        extra: Mapping[str, Any],
    ) -> bool:
        """Write the case files, validate with a full run, keep or discard."""
        case = self.tmp / f"case{len(self.cases):03d}"
        case.mkdir(parents=True, exist_ok=True)
        for name, write in files.items():
            write(case / name)
        book = case / "book_override.json" if "book_override.json" in files else None
        market = case / "market_override.parquet" if "market_override.parquet" in files else None
        doc = run_case(f"{day:%Y-%m-%d}", book, market, self.seed)
        alert = alert_of(doc["limits"])
        base_u = float(self.util.at[day, f"{alert['scope']}/{alert['metric']}"])
        why = check(kind, doc, base_u, extra)
        if why:
            if doc["run_id"] != f"{day:%Y-%m-%d}":
                shutil.rmtree(cli.RUNS / doc["run_id"], ignore_errors=True)
            shutil.rmtree(case)
            print(f"  reject {kind} {day:%Y-%m-%d}: {why}")
            return False
        self.used.add(day)
        self.cases.append(
            {"kind": kind, "day": day, "dir": case, "doc": doc, "alert": alert}
            | {"base_utilization": base_u, "extra": dict(extra)}
        )
        print(
            f"  accept {kind} {day:%Y-%m-%d} {alert['scope']}/{alert['metric']} "
            f"{alert['utilization']:.1%}"
        )
        return True

    # -- case types ------------------------------------------------------------------------
    def controls(self, n: int) -> None:
        near = round(n * NEAR_MISS_SHARE)
        mx = self.util.max(axis=1)
        for want, mask, near_miss in (
            (near, mx.between(0.91, 0.985), True),  # margins keep the status off the edges
            (n - near, mx < 0.88, False),
        ):
            got = 0
            for day in self.pool(mask):
                if got == want:
                    break
                got += self.accept("control", day, {}, {"near_miss": near_miss})
            if got < want:
                raise RuntimeError("not enough control dates")

    def market_shocks(self, n: int) -> None:
        got = 0
        for day in self.pool(self.util.max(axis=1) >= 1.0):
            if got == n:
                break
            got += self.accept("market_shock", day, {}, {})
        if got < n:
            raise RuntimeError("not enough clean breach days")

    def new_trade(self, day: pd.Timestamp, desk: str, kind: str) -> Trade:
        """One trade of `kind` struck on `day` by the book generator, numbered after the book."""
        cfg = cli.load_yaml("book")
        for d, specs in cfg["desks"].items():
            for k, spec in specs.items():
                spec["count"] = int(d == desk and k == kind)
        [trade], _ = generate_book(cfg, self.ctx.state(day), int(self.rng.integers(1 << 31)))
        n = sum(t.trade_id.startswith(PREFIX[kind]) for t in self.book)
        return trade.model_copy(update={"trade_id": f"{PREFIX[kind]}{n + 1:03d}"})

    def size(self, trade: Trade, day: pd.Timestamp, target: float) -> Trade | None:
        """Scale the new trade's notional until its desk's VaR utilization reaches `target`."""

        def util(t: Trade) -> float:
            return float(self.var_util(self.ctx, [*self.book, t], day)[t.desk])

        for flip in (False, True):
            t = trade
            if flip:
                t = (
                    t.model_copy(
                        update={
                            "pay_receive": "payer" if t.pay_receive == "receiver" else "receiver"
                        }
                    )
                    if t.instrument_type == "interest_rate_swap"
                    else t.model_copy(update={"notional": -t.notional})
                )
            lo, hi = 0.0, 1.0
            while util(t.model_copy(update={"notional": t.notional * hi})) < target:
                lo, hi = hi, hi * 2
                if hi > 64:
                    break
            else:
                for _ in range(20):  # bisection on the notional multiplier
                    mid = (lo + hi) / 2
                    x = t.model_copy(update={"notional": t.notional * mid})
                    lo, hi = (mid, hi) if util(x) < target else (lo, mid)
                notional = round(t.notional * hi / 5e5) * 5e5  # 0.5M lots
                return t.model_copy(update={"notional": notional})
        return None

    def position_jumps(self, n: int) -> None:
        desks = [list(KINDS)[i % len(KINDS)] for i in range(n)]
        mask = self.util.max(axis=1) < 1
        days = iter(self.pool(mask))
        for desk in desks:
            for _ in range(50):
                day = next(days)
                trade = self.new_trade(day, desk, str(self.rng.choice(KINDS[desk])))
                sized = self.size(trade, day, float(self.rng.uniform(1.08, 1.6)))
                if sized is None:
                    continue
                rows = [t.model_dump(mode="json") for t in [*self.book, sized]]

                def write(p: Path, rows: list[dict[str, Any]] = rows) -> None:
                    p.write_text(json.dumps({"trades": rows}, indent=1) + "\n")

                extra = {"trade": sized.model_dump(mode="json")}
                if self.accept("position_jump", day, {"book_override.json": write}, extra):
                    break
            else:
                raise RuntimeError(f"no position jump found for desk {desk}")

    def corrupted(self, day: pd.Timestamp, factor: str, how: str) -> tuple[float, float] | None:
        """(corrupted level, shock) for `factor` on `day`, or None if a sign flip is too small."""
        prev = self.ctx.levels.at[self.ctx.prev(day), factor]
        s = self.ctx.shocks[factor].loc[:day]
        vol = float(s.iloc[-61:-1].std())
        cap = 0.9 * float(self.cfgs.risk["controls"]["max_move"][factor])
        if how == "sign_flip":
            shock = -float(s.iloc[-1])
            if abs(shock) < 2.5 * vol:
                return None
        else:
            shock = float(self.rng.choice([-1, 1]) * self.rng.uniform(4, 12) * vol)
        shock = float(np.clip(shock, -cap, cap))
        level = prev + shock / 100 if factor in CURVE else prev * np.exp(shock)
        source = self.cfgs.risk["controls"]["cross_source"]["pairs"].get(factor)
        if source:  # stay inside the cross-source tolerance, or the rule check catches it
            ref = float(self.panel[source].shift(1).ffill(limit=3).loc[day])
            level = float(np.clip(level, ref * (1 - FX_GAP), ref * (1 + FX_GAP)))
            shock = float(np.log(level / prev))
        return float(level), shock

    def bad_data(self, n: int) -> None:
        """Scarce factors first (TRIES_PER_FACTOR each), then EURUSD=X fills the rest: only its
        corruptions reliably draw an anomaly warning and a breach together (ADR-010)."""
        cc = self.cfgs.risk["controls"]
        got: Counter[str] = Counter()
        order = [f for f in FACTORS if f != FILL_FACTOR] + [FILL_FACTOR]
        for factor in order:
            desk = DESK_OF[factor]
            near = self.util[f"{desk}/var_99_1d"].between(0.85, 0.9999)
            pool = self.pool(near & (self.util.max(axis=1) < 1))
            limit = MAX_TRIES if factor == FILL_FACTOR else TRIES_PER_FACTOR
            for day in pool[:limit]:
                if got.total() == n:
                    break
                how = str(self.rng.choice(["spike", "sign_flip"]))
                bad = self.corrupted(day, factor, how)
                if bad is None:
                    continue
                ov = pd.DataFrame({factor: [bad[0]]}, index=pd.DatetimeIndex([day], name="date"))
                panel = ov.combine_first(self.panel)
                recent = panel.loc[day - pd.offsets.BDay(400) : day]
                f = run_controls(recent, cc, self.forest, dates=[day])
                if (f["severity"] == "critical").any() or not (
                    (f["factor"] == factor) & (f["severity"] == "warning")
                ).any():
                    continue
                ctx = RiskContext.from_panel(panel, self.cfgs.risk)
                util = self.var_util(ctx, self.book, day)
                if not any(util[s] >= 1.01 > self.util.at[day, f"{s}/var_99_1d"] for s in util):
                    continue

                def write(p: Path, ov: pd.DataFrame = ov) -> None:
                    ov.to_parquet(p)

                extra = {"factor": factor, "corruption": how, "shock": bad[1], "level": bad[0]}
                extra |= {"clean_level": float(self.panel.at[day, factor])}
                if self.accept("bad_data", day, {"market_override.parquet": write}, extra):
                    got[factor] += 1
        print(f"  bad data by factor: {dict(got)}")
        if got.total() < n:
            raise RuntimeError("not enough bad-data cases; widen the search")

    # -- output ----------------------------------------------------------------------------
    def write(self) -> dict[str, Any]:
        """Assign shuffled IDs, move case files into place, write labels and the split."""
        order = self.rng.permutation(len(self.cases))
        final: dict[str, dict[str, Any]] = {}
        for i, j in enumerate(order):
            c = self.cases[j]
            iid = f"INC-{i + 1:03d}"
            dest = self.out / iid
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(c["dir"], dest)
            day = f"{c['day']:%Y-%m-%d}"
            files = ("book_override.json", "market_override.parquet")
            book, market = (f if (dest / f).exists() else None for f in files)
            run_id = cli.run_id_for(
                day, dest / book if book else None, dest / market if market else None
            )
            assert run_id == c["doc"]["run_id"]
            alert = c["alert"]
            incident: dict[str, Any] = {"incident_id": iid, "as_of_date": day, "run_id": run_id}
            incident["alert"] = {"scope": alert["scope"], "metric": alert["metric"]}
            incident |= {"book_override": book, "market_override": market}
            root, action = LABELS[c["kind"]]
            near = bool(c["extra"].get("near_miss", False))
            truth = {
                "incident_id": iid,
                "type": c["kind"],
                "near_miss": near,
                "as_of_date": day,
                "run_id": run_id,
                "breach": alert,
                "base_utilization": round(c["base_utilization"], 6),
                "root_cause": root,
                "recommended_action": action,
                "policy_sections": expected_sections(c["kind"], alert, near),
                "injection": c["extra"],
                "top_factors": top_factors(c["doc"], alert["scope"]),
                "dq_findings": c["doc"]["controls"]["findings"],
            }
            write_json(dest / "incident.json", incident)
            write_json(dest / "ground_truth.json", truth)
            final[iid] = truth
        shutil.rmtree(self.tmp, ignore_errors=True)
        return final


def write_json(path: Path, doc: Any) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")


def stratum(t: Mapping[str, Any]) -> str:
    return t["type"] + ("_near_miss" if t["near_miss"] else "")


def split(truth: Mapping[str, Mapping[str, Any]], seed: int) -> dict[str, list[str]]:
    """Stratified dev/test split by type (near misses as their own stratum)."""
    rng = np.random.default_rng(seed)
    strata: dict[str, list[str]] = {}
    for iid, t in sorted(truth.items()):
        strata.setdefault(stratum(t), []).append(iid)
    want = round(DEV_SHARE * len(truth))
    quota = {s: int(DEV_SHARE * len(ids)) for s, ids in strata.items()}
    rest = [s for s, ids in strata.items() if DEV_SHARE * len(ids) > quota[s]]
    for s in rng.choice(rest, want - sum(quota.values()), replace=False):
        quota[str(s)] += 1
    dev = sorted(i for s, ids in strata.items() for i in rng.choice(ids, quota[s], replace=False))
    return {"dev": dev, "test": sorted(set(truth) - set(dev))}


def split_checksum(out: Path, test_ids: Sequence[str]) -> str:
    """SHA-256 over the test IDs and their ground_truth.json bytes: the frozen test split."""
    h = hashlib.sha256()
    for iid in sorted(test_ids):
        h.update(iid.encode() + b"\n" + (out / iid / "ground_truth.json").read_bytes())
    return h.hexdigest()


def load_split(out: Path = OUT, pinned: str | None = None) -> dict[str, Any]:
    """split.json, after checking the frozen test split: the stored checksum, the recomputed one,
    and the one pinned in configs/agents.yaml must all agree."""
    doc: dict[str, Any] = json.loads((out / "split.json").read_text())
    got = split_checksum(out, doc["test"])
    if not got == doc["test_sha256"] == pinned:
        raise RuntimeError(
            f"test split changed: recomputed {got[:12]}, split.json {doc['test_sha256'][:12]}, "
            f"pinned {str(pinned)[:12]}"
        )
    return doc


def generate(n: int, seed: int, out: Path = OUT) -> dict[str, Any]:
    """Generate, validate, and write n incidents plus split.json. Returns the split document."""
    if n % 4:
        raise ValueError("n must be a multiple of 4 (equal counts per type)")
    g = Generator(seed, out)
    for kind, make in (
        ("control", g.controls),
        ("market_shock", g.market_shocks),
        ("position_jump", g.position_jumps),
        ("bad_data", g.bad_data),
    ):
        print(kind)
        make(n // 4)
    truth = g.write()
    s = split(truth, seed)
    doc = {"seed": seed, "n": n, "dev": s["dev"], "test": s["test"]}
    doc |= {"test_sha256": split_checksum(out, s["test"])}
    doc["strata"] = {part: dict(Counter(stratum(truth[i]) for i in ids)) for part, ids in s.items()}
    write_json(out / "split.json", doc)
    config = {"n": n, "seed": seed, "years": YEARS, "dev_share": DEV_SHARE}
    config |= {"near_miss_share": NEAR_MISS_SHARE, "tries_per_factor": TRIES_PER_FACTOR}
    write_json(out / "generate.config.json", config)
    return doc
