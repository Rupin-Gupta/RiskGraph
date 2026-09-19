"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge, Problem } from "@/components/ui";
import { type Incident, RUNNING, api, limitName } from "@/lib/api";

export default function IncidentsPage() {
  const [rows, setRows] = useState<Incident[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    const load = () =>
      api<Incident[]>("/incidents").then(
        (x) => {
          setRows(x);
          setError(null);
          if (x.some((i) => RUNNING.includes(i.status))) timer = setTimeout(load, 10_000);
        },
        (e: Error) => setError(e.message),
      );
    load();
    return () => clearTimeout(timer);
  }, []);

  return (
    <>
      <h1 className="text-xl font-semibold">Incidents</h1>
      <p className="text-sm text-ink-2">
        One per limit breach in a daily run. Investigations pause for your approval before any note is sent.
      </p>
      <Problem error={error} />
      {rows && rows.length === 0 && <p className="text-sm text-ink-2">No incidents yet.</p>}
      {rows && rows.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-line bg-surface">
          <table className="data w-full text-sm">
            <thead>
              <tr>
                <th>Incident</th>
                <th>As of</th>
                <th>Limit</th>
                <th>Status</th>
                <th>Updated</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((i) => (
                <tr key={i.incident_id}>
                  <td>
                    <Link className="underline" href={`/incidents/view?id=${encodeURIComponent(i.incident_id)}`}>
                      {i.incident_id}
                    </Link>
                  </td>
                  <td>{i.as_of_date}</td>
                  <td>{limitName(i)}</td>
                  <td>
                    <Badge status={i.status} />
                  </td>
                  <td className="text-ink-2">{new Date(i.updated_at).toLocaleString()}</td>
                  <td className="max-w-xs truncate text-ink-2" title={i.reason}>
                    {i.reason}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
