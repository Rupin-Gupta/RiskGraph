"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { type LimitRow, type Summary, label, limitName, pct, usd } from "@/lib/api";

const AXIS = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 }, tickLine: false };
const REF = { fill: "var(--ink-2)", fontSize: 11 };
const CURSOR = { fill: "var(--grid)", opacity: 0.5 };

type TipRow = { key: string; value: string; color?: string };

// Values lead, labels follow; series keyed by a short line, never by colored text.
function Tip({ title, rows }: { title: string; rows: TipRow[] }) {
  return (
    <div className="rounded border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <div className="mb-1 text-ink-2">{title}</div>
      {rows.map((r) => (
        <div key={r.key} className="flex items-center gap-2">
          {r.color && <span className="inline-block h-0.5 w-3" style={{ background: r.color }} />}
          <strong className="tabular-nums text-ink">{r.value}</strong>
          <span className="text-ink-2">{r.key}</span>
        </div>
      ))}
    </div>
  );
}

/** Utilization of each limit (one series), with the 90% warning and 100% limit lines. */
export function UtilizationChart({ rows }: { rows: LimitRow[] }) {
  const data = rows.map((r) => ({ ...r, name: limitName(r), util: r.utilization * 100 }));
  const max = Math.max(110, ...data.map((d) => Math.ceil(d.util / 10) * 10 + 10));
  return (
    <ResponsiveContainer width="100%" height={44 * data.length + 50}>
      <BarChart data={data} layout="vertical" margin={{ top: 18, right: 56, left: 8 }}>
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        <XAxis type="number" domain={[0, max]} unit="%" {...AXIS} />
        <YAxis type="category" dataKey="name" width={140} {...AXIS} />
        <ReferenceLine x={90} stroke="var(--axis)" label={{ value: "warning", position: "top", ...REF }} />
        <ReferenceLine x={100} stroke="var(--ink-2)" label={{ value: "limit", position: "top", ...REF }} />
        <Tooltip
          cursor={CURSOR}
          content={({ active, payload }) => {
            const d = active ? payload?.[0]?.payload : null;
            return d ? (
              <Tip
                title={d.name}
                rows={[
                  { key: "utilization", value: pct(d.utilization) },
                  { key: "value", value: usd(d.value) },
                  { key: "limit", value: usd(d.limit) },
                ]}
              />
            ) : null;
          }}
        />
        <Bar dataKey="util" fill="var(--series-1)" barSize={20} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          <LabelList
            dataKey="util"
            position="right"
            formatter={(v) => `${Number(v).toFixed(1)}%`}
            fill="var(--ink-2)"
            fontSize={12}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

const SERIES = [
  { key: "historical", name: "Historical simulation", color: "var(--series-1)" },
  { key: "monte_carlo", name: "Monte Carlo", color: "var(--series-2)" },
] as const;

/** 99% VaR exceptions per desk, by method, against the expected count. */
export function BacktestChart({ bt }: { bt: NonNullable<Summary["backtest"]> }) {
  const desks = Object.keys(bt.historical);
  const data = desks.map((d) => ({
    desk: label(d),
    historical: bt.historical[d].exceptions,
    monte_carlo: bt.monte_carlo[d]?.exceptions,
  }));
  const expected = bt.historical[desks[0]]?.expected;
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} barGap={2} margin={{ top: 18, right: 16 }}>
        <CartesianGrid vertical={false} stroke="var(--grid)" />
        <XAxis dataKey="desk" {...AXIS} />
        <YAxis allowDecimals={false} width={32} {...AXIS} />
        {expected !== undefined && (
          <ReferenceLine
            y={expected}
            stroke="var(--ink-2)"
            label={{ value: `expected ${expected}`, position: "insideTopRight", ...REF }}
          />
        )}
        <Tooltip
          cursor={CURSOR}
          content={({ active, payload, label: desk }) =>
            active && payload?.length ? (
              <Tip
                title={String(desk)}
                rows={SERIES.map((s) => ({
                  key: s.name,
                  color: s.color,
                  value: `${payload[0].payload[s.key]} exceptions`,
                }))}
              />
            ) : null
          }
        />
        <Legend
          iconType="rect"
          iconSize={10}
          formatter={(v) => <span style={{ color: "var(--ink-2)", fontSize: 12 }}>{v}</span>}
        />
        {SERIES.map((s) => (
          <Bar
            key={s.key}
            dataKey={s.key}
            name={s.name}
            fill={s.color}
            barSize={20}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
