import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RiskGraph",
  description: "Market risk limits and the incident approval queue for Meridian Bank (fictional).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <header className="border-b border-line bg-surface">
          <nav className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3 text-sm">
            <span className="font-semibold">RiskGraph</span>
            <Link href="/" className="text-ink-2 hover:text-ink">
              Risk
            </Link>
            <Link href="/incidents" className="text-ink-2 hover:text-ink">
              Incidents
            </Link>
            <span className="ml-auto text-xs text-muted">Meridian Bank (fictional) · synthetic data</span>
          </nav>
        </header>
        <main className="mx-auto max-w-6xl space-y-4 px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
