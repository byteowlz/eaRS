#!/usr/bin/env bun
// Cadence gate for an OWNED synthetic native-window replay, not scanout proof.
export function cadence(report: any, hz: number) {
  if (!Number.isFinite(hz) || hz < 30 || hz > 240) throw new Error("Refresh rate must be30..240Hz");
  for (const field of ["move_gap_p50_ms", "move_gap_p95_ms", "native_moves", "motion_seconds"]) {
    if (!Number.isFinite(report[field]) || report[field] <= 0) throw new Error(`Invalid motion field ${field}`);
  }
  if (report.ok !== true || report.motion_seconds < 3 || report.native_moves < 30) throw new Error("Incomplete native replay");
  const frame = 1000 / hz;
  return { pass: report.move_gap_p50_ms <= frame * 1.25 && report.move_gap_p95_ms <= frame * 1.5,
    refresh_hz: hz, median_budget_ms: frame * 1.25, p95_budget_ms: frame * 1.5 };
}
if (import.meta.main) {
  try {
    const [beforePath, afterPath, rate, ...extras] = process.argv.slice(2);
    if (!beforePath || !afterPath || !rate || extras.length) throw new Error("Usage: check_preview_motion.ts before.json after.json refresh_hz");
    const before = cadence(await Bun.file(beforePath).json(), Number(rate));
    const after = cadence(await Bun.file(afterPath).json(), Number(rate));
    const ok = !before.pass && after.pass;
    console.log(JSON.stringify({ ok, before, after, scope: "native window-bounds cadence; physical/high-refresh acceptance separate" }));
    if (!ok) process.exit(1);
  } catch (error) { console.error(String(error)); process.exit(2); }
}
