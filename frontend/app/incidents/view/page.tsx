"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { Badge, Card, Problem } from "@/components/ui";
import { type IncidentDetail, RUNNING, api, label, limitName, pct, usd } from "@/lib/api";

// Static export: the incident ID travels as ?id=, read on the client.
export default function Page() {
  return (
    <Suspense>
      <Detail />
    </Suspense>
  );
}

const DECIDABLE = ["awaiting_approval", "dispatch_failed"];

function Detail() {
  const id = useSearchParams().get("id") ?? "";
  const [d, setD] = useState<IncidentDetail | null>(null);
  const [note, setNote] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const show = useCallback((x: IncidentDetail) => {
    setD(x);
    setNote(x.report?.draft_note ?? "");
    setError(null);
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      api<IncidentDetail>(`/incidents/${encodeURIComponent(id)}`).then(
        (x) => {
          show(x);
          if (RUNNING.includes(x.status)) timer = setTimeout(load, 10_000);
        },
        (e: Error) => setError(e.message),
      );
    if (id) load();
    return () => clearTimeout(timer);
  }, [id, show]);

  async function decide(kind: "approve" | "reject") {
    if (!d) return;
    setBusy(true);
    const edited = note !== d.report?.draft_note ? note : null;
    try {
      show(await api<IncidentDetail>(`/incidents/${encodeURIComponent(id)}/${kind}`,
        kind === "approve" ? { edits: edited } : { reason }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!d) return <Problem error={error ?? (id ? null : "No incident ID in the URL.")} />;
  const r = d.report;
  const canDecide = DECIDABLE.includes(d.status) && r;
  return (
    <>
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold">{d.incident_id}</h1>
        <Badge status={d.status} />
        {d.trace_url && (
          <a className="ml-auto text-sm underline" href={d.trace_url} target="_blank" rel="noreferrer">
            Langfuse trace
          </a>
        )}
      </div>
      <p className="text-sm text-ink-2">
        {limitName(d)} limit, as of {d.as_of_date}
        {d.reason && ` · ${d.reason}`}
      </p>
      <Problem error={error} />

      {r && (
        <div className="grid gap-4 sm:grid-cols-4">
          {[
            ["Value", usd(r.breach.value)],
            ["Limit", usd(r.breach.limit)],
            ["Utilization", pct(r.breach.utilization)],
            ["Root cause", `${label(r.root_cause)} (${pct(r.confidence)})`],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg border border-line bg-surface p-3">
              <div className="text-xs text-ink-2">{k}</div>
              <div className="text-lg font-semibold">{v}</div>
            </div>
          ))}
        </div>
      )}

      {r && (
        <Card title="Escalation note" sub={`Recommended action: ${label(r.recommended_action)}.`}>
          {canDecide ? (
            <div className="space-y-3">
              <label className="block text-sm text-ink-2" htmlFor="note">
                Edit the note before approving; approval sends it as shown.
              </label>
              <textarea
                id="note"
                className="h-64 w-full rounded border border-line bg-page p-2 font-mono text-sm"
                value={note}
                onChange={(e) => setNote(e.target.value)}
              />
              <div className="flex flex-wrap items-center gap-2">
                <button
                  className="rounded bg-ink px-3 py-1.5 text-sm text-page disabled:opacity-50"
                  disabled={busy || !note.trim()}
                  onClick={() => decide("approve")}
                >
                  {d.status === "dispatch_failed" ? "Retry sending" : "Approve and send"}
                </button>
                <input
                  className="min-w-64 flex-1 rounded border border-line bg-page px-2 py-1.5 text-sm"
                  placeholder="Reason for rejecting"
                  aria-label="Reason for rejecting"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
                <button
                  className="rounded border border-line px-3 py-1.5 text-sm disabled:opacity-50"
                  disabled={busy || !reason.trim() || d.status === "dispatch_failed"}
                  onClick={() => decide("reject")}
                >
                  Reject
                </button>
              </div>
            </div>
          ) : (
            <pre className="whitespace-pre-wrap font-sans text-sm">{r.draft_note}</pre>
          )}
          {d.decision && (
            <p className="mt-3 text-sm text-ink-2">
              Decision: {d.decision.decision}
              {d.decision.reason && ` (${d.decision.reason})`}
              {d.note_path && ` · audit copy ${d.note_path}`}
            </p>
          )}
        </Card>
      )}

      {r && (
        <Card title="Evidence" sub="Each value is copied from a logged tool result; the critic re-fetched it by result ID.">
          <div className="overflow-x-auto">
            <table className="data w-full text-sm">
              <thead>
                <tr>
                  <th>Claim</th>
                  <th className="text-right">Value</th>
                  <th>Unit</th>
                  <th>Tool</th>
                  <th>Result ID</th>
                </tr>
              </thead>
              <tbody>
                {r.evidence.map((e) => (
                  <tr key={`${e.result_id}-${e.claim}`}>
                    <td>{e.claim}</td>
                    <td className="num">{e.value.toLocaleString("en-US", { maximumFractionDigits: 6 })}</td>
                    <td>{e.unit}</td>
                    <td>{e.tool ?? "–"}</td>
                    <td className="font-mono text-xs">{e.result_id}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {r && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Policy citations">
            <ul className="list-inside list-disc text-sm">
              {r.policy_citations.map((c) => (
                <li key={`${c.doc_id}-${c.section_id}`}>
                  {c.doc_id} <span className="font-mono">{c.section_id}</span>
                </li>
              ))}
            </ul>
          </Card>
          {d.critique && (
            <Card title="Critic">
              <p className="text-sm">
                <Badge status={d.critique.passed ? "ok" : "warning"} />{" "}
                {d.critique.passed ? "All checks passed" : "Open issues"} after {d.critique.loops} revision loop(s).
              </p>
              <ul className="mt-2 list-inside list-disc text-sm text-ink-2">
                {d.critique.issues.map((i) => (
                  <li key={i.message}>
                    {i.agent}: {i.message}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
      {!r && RUNNING.includes(d.status) && (
        <p className="text-sm text-ink-2">The agents are investigating; this page refreshes every 10 seconds.</p>
      )}
    </>
  );
}
