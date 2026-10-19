import type { ReactNode } from "react";

export function Panel({
  title,
  children,
  className = "",
  right,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}) {
  return (
    <section className={`panel ${className}`}>
      {title && (
        <header className="mb-3 flex items-center justify-between gap-2">
          <h2 className="panel-title">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatCard({
  label,
  value,
  sub,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  sub?: string;
  tone?: "neutral" | "green" | "amber" | "red" | "blue";
}) {
  const toneClass = {
    neutral: "text-slate-100",
    green: "text-signal-green",
    amber: "text-signal-amber",
    red: "text-signal-red",
    blue: "text-signal-blue",
  }[tone];
  return (
    <div className="panel flex flex-col justify-between">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-2 font-mono text-2xl font-semibold ${toneClass}`}>{value}</p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

export function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    low: "border-signal-green/40 bg-signal-green/10 text-signal-green",
    medium: "border-signal-amber/40 bg-signal-amber/10 text-signal-amber",
    high: "border-signal-red/40 bg-signal-red/10 text-signal-red",
  };
  return (
    <span
      className={`inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${
        styles[risk] ?? "border-navy-600 bg-navy-800 text-slate-400"
      }`}
    >
      {risk} risk
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-400" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-navy-600 border-t-signal-blue" />
      {label}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-signal-red/40 bg-signal-red/10 px-4 py-3 text-sm text-signal-red"
    >
      <strong className="font-semibold">Error: </strong>
      {message}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-dashed border-navy-600 px-4 py-8 text-center text-sm text-slate-500">
      {message}
    </div>
  );
}

export function MetricBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 1000) / 10;
  const color =
    value >= 0.8 ? "bg-signal-green" : value >= 0.5 ? "bg-signal-amber" : "bg-signal-red";
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded bg-navy-800">
        <div className={`h-full rounded ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
