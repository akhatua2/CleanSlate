// Shared helpers for painting per-word memory onto lyrics.
export const STEPS = [-3, -2.5, -2, -1.5, -1, -0.5]; // lower edges of the memory bins (log10 p_z)

// v = round(log10 p_z * 10), or null when below the 1e-3 extraction threshold
export function memLevel(v) {
  if (v == null) return 0;
  const lp = v / 10;
  let i = 0;
  while (i < STEPS.length - 1 && lp >= STEPS[i + 1]) i++;
  return i + 2; // 2..7, matching --mem-2..--mem-7
}

export function fmtP(v) {
  if (v == null) return "not extractable (< 10⁻³)";
  const lp = v / 10;
  if (lp >= -0.05) return "≈ 1";
  const p = Math.pow(10, lp);
  return p >= 0.01 ? p.toFixed(2) : `10^${lp.toFixed(1)}`;
}
