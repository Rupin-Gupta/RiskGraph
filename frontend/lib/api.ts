// Client for the FastAPI service. nginx proxies /api to it, so the pages stay same-origin.
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api";

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    const detail = await r.json().then((j) => j.detail, () => r.statusText);
    throw new Error(`${r.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return r.json() as Promise<T>;
}

export type LimitRow = {
  scope: string;
  metric: string;
  value: number;
  limit: number;
  utilization: number;
  status: string;
};

export type BacktestDesk = {
  exceptions: number;
  expected: number;
  kupiec_p_value: number;
  traffic_light: string;
};

export type Summary = {
  date: string;
  dates: string[];
  desks: Record<string, Record<string, number>>;
  limits: LimitRow[];
  backtest: {
    start: string;
    end: string;
    days: number;
    historical: Record<string, BacktestDesk>;
    monte_carlo: Record<string, BacktestDesk>;
  } | null;
};

export type Incident = {
  incident_id: string;
  thread_id: string;
  as_of_date: string;
  run_id: string;
  scope: string;
  metric: string;
  status: string;
  reason: string;
  created_at: string;
  updated_at: string;
};

export type Evidence = { claim: string; value: number; unit: string; result_id: string; tool: string | null };

export type Report = {
  breach: { scope: string; metric: string; value: number; limit: number; utilization: number };
  root_cause: string;
  confidence: number;
  evidence: Evidence[];
  recommended_action: string;
  policy_citations: { doc_id: string; section_id: string }[];
  draft_note: string;
};

export type IncidentDetail = Incident & {
  report: Report | null;
  critique: { passed: boolean; loops: number; issues: { agent: string; message: string }[] } | null;
  decision: { decision: string; reason?: string } | null;
  note_path: string | null;
  trace_url: string | null;
};

export const RUNNING = ["queued", "investigating"]; // statuses the pages poll on

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 });
export const usd = (x: number) => `USD ${compact.format(x)}`;
export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const label = (s: string) => s.replaceAll("_", " ");
export const limitName = (r: { scope: string; metric: string }) =>
  `${label(r.scope)} ${r.metric === "var_99_1d" ? "VaR" : label(r.metric)}`;
