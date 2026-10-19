import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  ScanSearch,
  FileSpreadsheet,
  LineChart,
  Info,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "../services/api";
import type { Health } from "../types";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyzer", label: "Transaction Analyzer", icon: ScanSearch },
  { to: "/batch", label: "Batch Analysis", icon: FileSpreadsheet },
  { to: "/performance", label: "Model Performance", icon: LineChart },
  { to: "/about", label: "About Project", icon: Info },
];

export default function Layout({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const modelReady = health?.model_ready ?? false;

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      {/* Sidebar */}
      <aside className="border-b border-navy-700 bg-navy-900 lg:w-60 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2 px-4 py-4">
          <ShieldCheck className="h-7 w-7 text-signal-blue" aria-hidden />
          <div>
            <h1 className="text-base font-bold tracking-tight text-slate-100">
              FraudShield <span className="text-signal-blue">AI</span>
            </h1>
            <p className="text-[11px] text-slate-500">Transaction Risk Workstation</p>
          </div>
        </div>
        <nav aria-label="Main navigation" className="flex flex-row gap-1 overflow-x-auto px-2 pb-2 lg:flex-col lg:pb-4">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 whitespace-nowrap rounded-md px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? "bg-navy-700 font-medium text-signal-blue"
                    : "text-slate-400 hover:bg-navy-850 hover:text-slate-200"
                }`
              }
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden border-t border-navy-700 px-4 py-3 lg:block">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                modelReady ? "bg-signal-green" : "bg-signal-red"
              }`}
              aria-hidden
            />
            <span className="text-slate-400">
              {health === null
                ? "Checking API…"
                : modelReady
                ? `Model ready · v${health.model_version}`
                : "Model not loaded"}
            </span>
          </div>
          {health?.is_demo && (
            <p className="mt-2 rounded border border-signal-amber/40 bg-signal-amber/10 p-2 text-[11px] leading-snug text-signal-amber">
              DEMO MODE — trained on synthetic data. Metrics are not real-world
              fraud performance.
            </p>
          )}
        </div>
      </aside>

      {/* Content */}
      <main className="flex-1 overflow-x-hidden p-4 lg:p-6">{children}</main>
    </div>
  );
}
