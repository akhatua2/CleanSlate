"""Merge bbh_* and mmlu_* subtasks in the validation JSONs into single
bbh / mmlu accuracy deltas, plus a clean per-benchmark table for everything else.

Reads:
  - results/pre_val/llama-8b/validate/llama-3_1-8b.json
  - results/post_val/llama-8b-simnpo-baseline/validate/checkpoint-134.json

Reports BOTH equal-weighted (mean of per-subtask accuracies — what papers
usually quote for BBH/MMLU) and sample-weighted (sum num_correct / sum n)
aggregates, since they can disagree on imbalanced suites.
"""
import json
import sys
from pathlib import Path

PRE_PATH = Path("results/pre_val/llama-8b/validate/llama-3_1-8b.json")
POST_PATH = Path("results/post_val/llama-8b-simnpo-baseline/validate/checkpoint-134.json")
OUT_PATH = Path("results/post_val/llama-8b-simnpo-baseline/merged_deltas.json")


def load(path: Path) -> dict[str, dict]:
    return json.load(path.open())["benchmarks"]


def aggregate(b: dict[str, dict], prefix: str) -> tuple[float, float, int, int]:
    """Return (equal_weighted_acc, sample_weighted_acc, n_subtasks, total_samples)."""
    subs = [v for k, v in b.items() if k.startswith(prefix)]
    if not subs:
        return 0.0, 0.0, 0, 0
    eq = sum(s["accuracy"] for s in subs) / len(subs)
    total = sum(s["num_total"] for s in subs)
    sw = sum(s["accuracy"] * s["num_total"] for s in subs) / total if total else 0.0
    return eq, sw, len(subs), total


def main() -> None:
    pre = load(PRE_PATH)
    post = load(POST_PATH)

    rows: list[dict] = []

    for label, prefix in (("bbh", "bbh_cot_fewshot_"), ("mmlu", "mmlu_")):
        pe, ps, n_sub, n_samp = aggregate(pre, prefix)
        qe, qs, _, _ = aggregate(post, prefix)
        rows.append({
            "benchmark": label,
            "n_subtasks": n_sub,
            "n_samples": n_samp,
            "pre_eq":  pe, "post_eq":  qe, "delta_eq":  qe - pe,
            "pre_sw":  ps, "post_sw":  qs, "delta_sw":  qs - ps,
            "retention_eq": qe / pe if pe > 0 else 0.0,
            "retention_sw": qs / ps if ps > 0 else 0.0,
        })

    # Standalone benchmarks (everything that's not bbh_* or mmlu_*).
    standalone = sorted(set(pre) & set(post) - {
        k for k in (set(pre) | set(post))
        if k.startswith("bbh_cot_fewshot_") or k.startswith("mmlu_")
    })
    standalone_rows = []
    for name in standalone:
        pa, qa = pre[name]["accuracy"], post[name]["accuracy"]
        standalone_rows.append({
            "benchmark": name,
            "pre": pa, "post": qa,
            "delta": qa - pa,
            "retention": qa / pa if pa > 0 else 0.0,
            "n": pre[name]["num_total"],
        })

    # Combined view: standalone + merged bbh + merged mmlu using equal-weighted.
    combined_eq = standalone_rows + [
        {"benchmark": r["benchmark"], "pre": r["pre_eq"], "post": r["post_eq"],
         "delta": r["delta_eq"], "retention": r["retention_eq"],
         "n": r["n_samples"], "n_subtasks": r["n_subtasks"]}
        for r in rows
    ]
    combined_eq.sort(key=lambda x: x["benchmark"])

    utility_eq = sum(r["retention"] for r in combined_eq) / len(combined_eq)
    utility_sw = sum(r["retention"] for r in combined_eq) / len(combined_eq)  # placeholder
    # Sample-weighted utility: weight each row by its sample count.
    total_n = sum(r["n"] for r in combined_eq) or 1
    utility_sw_samples = sum(r["retention"] * r["n"] for r in combined_eq) / total_n

    out = {
        "merged_groups": rows,
        "standalone": standalone_rows,
        "combined_equal_weighted": combined_eq,
        "utility_retained_equal_weighted_over_benchmarks": utility_eq,
        "utility_retained_sample_weighted_over_benchmarks": utility_sw_samples,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # Pretty print to stderr.
    print(f"\n{'benchmark':32s}  {'pre':>7s}  {'post':>7s}  {'delta':>8s}  {'retent':>7s}  {'n':>6s}",
          file=sys.stderr)
    print("-" * 72, file=sys.stderr)
    for r in combined_eq:
        n_str = f"{r['n']}" + (f" ({r['n_subtasks']} sub)" if "n_subtasks" in r else "")
        print(f"{r['benchmark']:32s}  {r['pre']:7.4f}  {r['post']:7.4f}  "
              f"{r['delta']:+.4f}  {r['retention']:7.4f}  {n_str}", file=sys.stderr)
    print(f"\nutility_retained (equal-weighted across benchmarks): {utility_eq:.4f}", file=sys.stderr)
    print(f"utility_retained (sample-weighted across benchmarks): {utility_sw_samples:.4f}", file=sys.stderr)
    print(f"\nMerged BBH/MMLU details (equal vs sample weighted):", file=sys.stderr)
    for r in rows:
        print(f"  {r['benchmark']:6s} eq:  pre={r['pre_eq']:.4f} post={r['post_eq']:.4f} "
              f"Δ={r['delta_eq']:+.4f}  ({r['n_subtasks']} subtasks)", file=sys.stderr)
        print(f"         sw:  pre={r['pre_sw']:.4f} post={r['post_sw']:.4f} "
              f"Δ={r['delta_sw']:+.4f}  ({r['n_samples']} samples)", file=sys.stderr)
    print(f"\nWritten to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
