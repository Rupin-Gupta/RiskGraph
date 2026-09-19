"use client";

import { useEffect, useState } from "react";
import { BacktestChart, UtilizationChart } from "@/components/charts";
import { Badge, Card, Problem } from "@/components/ui";
import { type Summary, api, label, limitName, pct, usd } from "@/lib/api";

const METHODS = [
  ["var_99_1d_hs", "VaR 99% HS"],
  ["var_99_1d_mc", "VaR 99% MC"],
  ["var_99_1d_param", "VaR 99% parametric"],
  ["es_975_1d_hs", "ES 97.5% HS"],
] as const;

export default function RiskPage() {
  const [date, setDate] = useState("");
  const [s, setS] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Summary>(`/risk/summary${date ? `?date=${date}` : ""}`).then(
      (x) => (setS(x), setError(null)),
      (e: Error) => setError(e.message),
    );
  }, [date]);

  const firm = s?.limits.find((r) => r.scope === "firm" && r.metric === "var_99_1d");
  const bt = s?.backtest;
  return (
    <>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">Market risk</h1>
        {s && (
          <label className="ml-auto flex items-center gap-2 text-sm text-ink-2">
            As of
            <select
              className="rounded border border-line bg-surface px-2 py-1 text-ink"
              value={s.date}
              onChange={(e) => setDate(e.target.value)}
            >
              {[...s.dates].reverse().map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <Problem error={error} />
      {s && (
        <div className={error ? "opacity-50" : ""}>
          {firm && (
            <div className="mb-4">
              <div className="text-sm text-ink-2">Firm 99% 1-day VaR (historical simulation)</div>
              <div className="flex flex-wrap items-baseline gap-3">
                <span className="text-5xl font-semibold">{usd(firm.value)}</span>
                <span className="text-ink-2">
                  {pct(firm.utilization)} of the {usd(firm.limit)} limit
                </span>
                <Badge status={firm.status} />
              </div>
            </div>
          )}
          <div className="space-y-4">
            <Card title="Limit utilization" sub={`Each limit's metric as a share of the limit, ${s.date}.`}>
              <UtilizationChart rows={s.limits} />
              <table className="data mt-2 w-full text-sm">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="text-right">Value</th>
                    <th className="text-right">Limit</th>
                    <th className="text-right">Utilization</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {s.limits.map((r) => (
                    <tr key={`${r.scope}/${r.metric}`}>
                      <td>{limitName(r)}</td>
                      <td className="num">{usd(r.value)}</td>
                      <td className="num">{usd(r.limit)}</td>
                      <td className="num">{pct(r.utilization)}</td>
                      <td>
                        <Badge status={r.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="VaR and ES by method" sub="1-day, USD. Limits apply to 99% HS VaR.">
              <table className="data w-full text-sm">
                <thead>
                  <tr>
                    <th>Desk</th>
                    {METHODS.map(([k, name]) => (
                      <th key={k} className="text-right">
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(s.desks).map(([desk, m]) => (
                    <tr key={desk}>
                      <td>{label(desk)}</td>
                      {METHODS.map(([k]) => (
                        <td key={k} className="num">
                          {m[k] === undefined ? "–" : usd(m[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
          {bt && (
            <div className="mt-4">
              <Card
                title="VaR backtest exceptions"
                sub={`Days the hypothetical P&L loss exceeded 99% VaR, ${bt.start} to ${bt.end} (${bt.days} days; a fixed window, not the date above).`}
              >
                <BacktestChart bt={bt} />
                <table className="data mt-2 w-full text-sm">
                  <thead>
                    <tr>
                      <th>Desk</th>
                      <th className="text-right">HS exceptions</th>
                      <th>HS zone</th>
                      <th className="text-right">MC exceptions</th>
                      <th>MC zone</th>
                      <th className="text-right">Expected</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(bt.historical).map(([desk, h]) => (
                      <tr key={desk}>
                        <td>{label(desk)}</td>
                        <td className="num">{h.exceptions}</td>
                        <td>
                          <Badge status={h.traffic_light} />
                        </td>
                        <td className="num">{bt.monte_carlo[desk]?.exceptions}</td>
                        <td>
                          <Badge status={bt.monte_carlo[desk]?.traffic_light ?? "–"} />
                        </td>
                        <td className="num">{h.expected}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>
          )}
        </div>
      )}
    </>
  );
}
