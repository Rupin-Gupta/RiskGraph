import type { ReactNode } from "react";

// Status never rides on color alone: every badge pairs a colored icon with a text label.
const STATUS: Record<string, [string, string]> = {
  ok: ["good", "✓"],
  green: ["good", "✓"],
  dispatched: ["good", "✓"],
  warning: ["warning", "▲"],
  yellow: ["warning", "▲"],
  awaiting_approval: ["warning", "●"],
  needs_human: ["serious", "!"],
  breach: ["critical", "✕"],
  red: ["critical", "✕"],
  failed: ["critical", "✕"],
  dispatch_failed: ["critical", "✕"],
};

export function Badge({ status }: { status: string }) {
  const [tone, icon] = STATUS[status] ?? ["muted", "–"];
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs whitespace-nowrap">
      <span aria-hidden style={{ color: `var(--${tone})` }}>
        {icon}
      </span>
      {status.replaceAll("_", " ")}
    </span>
  );
}

export function Card({ title, sub, children }: { title: string; sub?: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-surface p-4">
      <h2 className="font-medium">{title}</h2>
      {sub && <p className="mb-3 text-sm text-ink-2">{sub}</p>}
      {children}
    </section>
  );
}

export function Problem({ error }: { error: string | null }) {
  return error ? (
    <p role="alert" className="rounded border border-line p-3 text-sm">
      <span aria-hidden style={{ color: "var(--critical)" }}>✕ </span>
      {error}
    </p>
  ) : null;
}
