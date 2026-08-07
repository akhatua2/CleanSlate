"""
KS battery for the CPT calibration experiment.

Inputs (parquet eval datasets, all with the per-window schema):
  --base    Llama-3.1-8B base eval on the post-cutoff dataset (HF or local).
  --cpt     Llama-CPT eval on the post-cutoff dataset (local).
  --unlearn Llama-CPT-unlearned eval on the post-cutoff dataset (local).
  --selection JSON listing target_content_ids (50) and heldout_content_ids (20).

Computes six KS distances on per-window p_z arrays:
  S1: base   vs cpt     on target songs   (sanity gate, expect large)
  S2: base   vs cpt     on held-out songs (sanity gate, expect small)
  P1: base   vs unlearn on target songs   (calibration result)
  P2: cpt    vs unlearn on target songs   (unlearning magnitude)
  SE1: cpt   vs unlearn on held-out songs (collateral on never-targeted)
  SE2: base  vs unlearn on held-out songs (residual after unlearning)

Reports each KS with two CIs:
  - window-level bootstrap (1000 iter; tight)
  - cluster-level bootstrap over songs (1000 iter; honest about which songs we picked)

Outputs:
  - results/cpt_calibration/ks_table.json
  - results/plots/cpt_calibration_target.{pdf,png}
  - results/plots/cpt_calibration_heldout.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_VERBOSITY", "error")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402


def _load_eval(spec: str) -> pl.DataFrame:
    """Load an eval dataset (HF id or local path) and return a per-window DataFrame.

    The window column is a list-of-structs; we explode it and grab `p_z` plus the
    parent `content_id` so we can filter per song subset.
    """
    if spec.startswith("hf://") or "/" in spec and not Path(spec).exists():
        # HF dataset id
        df = pl.read_parquet(f"hf://datasets/{spec}/data/train-*.parquet")
    else:
        path = Path(spec)
        if path.is_dir():
            # cleanslate eval default writes results.parquet under <out>/<config>/<model>/
            candidates = sorted(path.glob("**/results.parquet"))
            if not candidates:
                raise FileNotFoundError(f"no results.parquet under {path}")
            df = pl.read_parquet(str(candidates[0]))
        else:
            df = pl.read_parquet(str(path))
    if "windows" not in df.columns:
        raise ValueError(f"{spec}: expected per-window 'windows' column, got {df.columns}")
    out = (
        df.select(["content_id", "windows"])
        .explode("windows")
        .with_columns(pl.col("windows").struct.field("p_z").alias("p_z"))
        .select(["content_id", "p_z"])
    )
    return out


def _filter_ids(df: pl.DataFrame, ids: set[str]) -> pl.DataFrame:
    return df.filter(pl.col("content_id").is_in(list(ids)))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    res = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
    return float(res.statistic)


def _bootstrap_ks_window(a: np.ndarray, b: np.ndarray, n_iter: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats_list = []
    for _ in range(n_iter):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        stats_list.append(_ks(a_s, b_s))
    return float(np.percentile(stats_list, 2.5)), float(np.percentile(stats_list, 97.5))


def _bootstrap_ks_cluster(
    df_a: pl.DataFrame, df_b: pl.DataFrame, ids: list[str], n_iter: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Cluster-bootstrap: resample songs (clusters) with replacement, recompute KS on
    the resampled songs' windows. More conservative than window-level bootstrap."""
    rng = np.random.default_rng(seed)
    a_by_id = {cid: df_a.filter(pl.col("content_id") == cid)["p_z"].to_numpy() for cid in ids}
    b_by_id = {cid: df_b.filter(pl.col("content_id") == cid)["p_z"].to_numpy() for cid in ids}
    stats_list = []
    for _ in range(n_iter):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        a_sample = np.concatenate([a_by_id[cid] for cid in sample_ids])
        b_sample = np.concatenate([b_by_id[cid] for cid in sample_ids])
        stats_list.append(_ks(a_sample, b_sample))
    return float(np.percentile(stats_list, 2.5)), float(np.percentile(stats_list, 97.5))


def _summary(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _plot_cdf(states: dict[str, np.ndarray], title: str, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 12,
    })
    colors = {"base": "#7AA9D6", "cpt": "#E07B6B", "unlearn": "#7AC1A8"}
    labels = {
        "base": "Llama-base (never-seen reference)",
        "cpt": "Llama-CPT (memorized)",
        "unlearn": "Llama-CPT-unlearned",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    for state, arr in states.items():
        if len(arr) == 0:
            continue
        # Clip to a sensible log range; avoid log(0) by setting a floor.
        arr = np.clip(arr, 1e-40, 1.0)
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, color=colors[state], linewidth=3.0,
                alpha=0.9, label=labels[state])
    ax.set_xscale("log")
    ax.set_xlabel("$p_z$ (log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(title, pad=10)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        png = out.with_suffix(".png")
        fig.savefig(png, bbox_inches="tight", dpi=200)
        print(f"saved {out} and {png}")
    else:
        print(f"saved {out}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="HF dataset id or local path for base eval")
    ap.add_argument("--cpt", required=True, help="local path for Llama-CPT eval")
    ap.add_argument("--unlearn", required=True, help="local path for Llama-CPT-unlearned eval")
    ap.add_argument("--selection", required=True, help="path to selection.json")
    ap.add_argument("--out-dir", type=Path, default=Path("results/cpt_calibration"))
    ap.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--bootstrap-iter", type=int, default=1000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(Path(args.selection).read_text())
    target_ids = set(selection["target_content_ids"])
    heldout_ids = set(selection["heldout_content_ids"])
    sys.stderr.write(f"target={len(target_ids)} heldout={len(heldout_ids)}\n")

    base_df = _load_eval(args.base)
    cpt_df = _load_eval(args.cpt)
    unl_df = _load_eval(args.unlearn)

    # Per-state per-subset arrays
    states = {"base": base_df, "cpt": cpt_df, "unlearn": unl_df}
    target = {k: _filter_ids(d, target_ids)["p_z"].to_numpy() for k, d in states.items()}
    heldout = {k: _filter_ids(d, heldout_ids)["p_z"].to_numpy() for k, d in states.items()}

    # Six KS tests
    KS_DEFS = [
        ("S1",  "base",    "cpt",     "target songs",  "sanity: CPT injected memorization"),
        ("S2",  "base",    "cpt",     "heldout songs", "sanity: CPT did not generalize"),
        ("P1",  "base",    "unlearn", "target songs",  "primary: F=100% returns to never-seen?"),
        ("P2",  "cpt",     "unlearn", "target songs",  "magnitude of unlearning effect"),
        ("SE1", "cpt",     "unlearn", "heldout songs", "leakage on never-targeted content"),
        ("SE2", "base",    "unlearn", "heldout songs", "residual heldout shift after unlearning"),
    ]

    results: list[dict] = []
    for name, a_state, b_state, subset_label, descr in KS_DEFS:
        a_dict = target if subset_label == "target songs" else heldout
        a = a_dict[a_state]
        b = a_dict[b_state]
        ks = _ks(a, b)
        # Cluster bootstrap requires per-song splits.
        ids_for_subset = sorted(target_ids if subset_label == "target songs" else heldout_ids)
        df_a = _filter_ids(states[a_state], target_ids if subset_label == "target songs" else heldout_ids)
        df_b = _filter_ids(states[b_state], target_ids if subset_label == "target songs" else heldout_ids)
        cl_lo, cl_hi = _bootstrap_ks_cluster(df_a, df_b, ids_for_subset, args.bootstrap_iter)
        wn_lo, wn_hi = _bootstrap_ks_window(a, b, args.bootstrap_iter)
        results.append(
            {
                "name": name,
                "description": descr,
                "states": f"{a_state} vs {b_state}",
                "subset": subset_label,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
                "ks": ks,
                "ci_window_95": [wn_lo, wn_hi],
                "ci_cluster_95": [cl_lo, cl_hi],
            }
        )

    summary_per_state = {
        "target": {k: _summary(v) for k, v in target.items()},
        "heldout": {k: _summary(v) for k, v in heldout.items()},
    }

    output = {
        "ks_tests": results,
        "summary": summary_per_state,
        "criterion": "KS < 0.10 = match (pre-registered)",
        "n_target_songs": len(target_ids),
        "n_heldout_songs": len(heldout_ids),
    }
    out_json = args.out_dir / "ks_table.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_json}")

    # CDF plots
    _plot_cdf(target, "$p_z$ on the 50 target songs (CPT'd)",
              args.plot_dir / "cpt_calibration_target.pdf")
    _plot_cdf(heldout, "$p_z$ on the 20 held-out songs (never CPT'd)",
              args.plot_dir / "cpt_calibration_heldout.pdf")

    # Console table
    print("\n" + "=" * 95)
    print(f"{'Name':<5} {'States':<22} {'Subset':<15} {'KS':>6} {'CI(window)':>20} {'CI(cluster)':>20}")
    print("-" * 95)
    for r in results:
        ks = r["ks"]
        wlo, whi = r["ci_window_95"]
        clo, chi = r["ci_cluster_95"]
        match = "" if np.isnan(ks) else (" ✓match" if ks < 0.10 else " ✗")
        print(f"{r['name']:<5} {r['states']:<22} {r['subset']:<15} {ks:>6.3f} "
              f"[{wlo:>5.3f},{whi:>5.3f}]    [{clo:>5.3f},{chi:>5.3f}]{match}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
