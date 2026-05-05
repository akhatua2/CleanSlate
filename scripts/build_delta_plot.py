"""
Trajectory plots in the same 3-panel style as `pareto.pdf`:

  scripts/build_delta_plot.py             → both plots
  scripts/build_delta_plot.py --kind ckpt → only the checkpoint trajectory
  scripts/build_delta_plot.py --kind algo → only the algorithm comparison

Each plot has 3 panels (F%, R%, Mean Δ% on retention benchmarks) with one thick
diamond-marker line per model and a red dashed zero/ideal-state reference.

Run after `build_tables_from_hf.py --warm` so the HF cache is populated.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tables_from_hf as B  # noqa: E402

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared styling ────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Llama-3.1-8B": "#7AA9D6",   # dusty blue
    "Olmo-3-7B":    "#F4A992",   # peach
    "Qwen3-8B":     "#8AC57A",   # sage green
}


# ── Generic data helpers ──────────────────────────────────────────────────────
def _gen_mean(repo: str, config: str) -> float | None:
    payload = B._load_cache(repo, config)
    if not payload:
        return None
    if payload["kind"] == "mean":
        return payload["mean"]
    if payload["kind"] == "gsm8k":
        flex = [r["score"] for r in payload["rows"] if r.get("filter") == "flexible-extract"]
        return sum(flex) / len(flex) if flex else None
    if payload["kind"] == "cleanslate_qa":
        s = [r["score"] for r in payload["rows"]]
        return sum(s) / len(s) if s else None
    return None


def _bbh_mean(repo: str) -> float | None:
    files = sorted(B.CACHE_DIR.glob(f"{B._safe_name(repo)}__bbh_cot_fewshot_*.json"))
    means = []
    for f in files:
        sub = f.name.split("__")[-1].removesuffix(".json")
        m = _gen_mean(repo, sub)
        if m is not None:
            means.append(m)
    return sum(means) / len(means) if means else None


def _retention_avg_delta_pp(pre_gen: str, post_gen: str) -> float | None:
    """Mean Δpp on the 7 retention benchmarks (mirrors the AvgVal column)."""
    deltas = []
    for label, key in B.VAL_COLS:
        if key == "bbh":
            pre, post = _bbh_mean(pre_gen), _bbh_mean(post_gen)
        elif key == "gsm8k_flexible_extract":
            pre, post = _gen_mean(pre_gen, "gsm8k"), _gen_mean(post_gen, "gsm8k")
        else:
            pre, post = _gen_mean(pre_gen, key), _gen_mean(post_gen, key)
        if pre is not None and post is not None:
            deltas.append((post - pre) * 100)
    return (sum(deltas) / len(deltas)) if deltas else None


def _mem_split(eval_repo: str, forget_ids: set[str]) -> tuple[int, int, int, int] | None:
    """Returns (forget_n_w, forget_mem, retain_n_w, retain_mem)."""
    per = B._cache_eval(eval_repo)
    if not per:
        return None
    all_ids = set(per.keys())
    retain_ids = all_ids - forget_ids
    fW = sum(per[c]["num_windows"] for c in forget_ids if c in per)
    fM = sum(per[c]["memorized_windows"] for c in forget_ids if c in per)
    rW = sum(per[c]["num_windows"] for c in retain_ids if c in per)
    rM = sum(per[c]["memorized_windows"] for c in retain_ids if c in per)
    return fW, fM, rW, rM


# ── Stage configuration ───────────────────────────────────────────────────────
# Checkpoint trajectory: Base → 33% → 66% → 100% → Aggressive.
# All paths live in the "fsid-curated-{short}-target-100:forget:baseline" pool.
CKPT_STAGES = ["Base", "33%", "66%", "100%", "Aggressive"]

CKPT_PATHS: dict[str, dict[str, dict[str, str | None]]] = {
    "Llama-3.1-8B": {
        "Base":      {"eval": "unlearning-cleanslate/eval-llama-3_1-8b",
                      "gen":  "unlearning-cleanslate/generations-llama-3_1-8b-pre_val"},
        "33%":       {"eval": "unlearning-cleanslate/eval-checkpoint-67-w100-s10",
                      "gen":  "arpandeepk/generations-llama-3_1-8b-simnpo-gentle-checkpoint-67"},
        "66%":       {"eval": "unlearning-cleanslate/eval-checkpoint-134-w100-s10",
                      "gen":  "arpandeepk/generations-llama-3_1-8b-simnpo-gentle-checkpoint-134"},
        "100%":      {"eval": "unlearning-cleanslate/eval-checkpoint-201-w100-s10",
                      "gen":  "arpandeepk/generations-llama-3_1-8b-simnpo-gentle-checkpoint-201"},
        "Aggressive":{"eval": "unlearning-cleanslate/eval-05-llama-3_1-8b-simnpo-baseline-target-100-checkpoint-432",
                      "gen":  "unlearning-cleanslate/generations-05-llama-3_1-8b-simnpo-baseline-target-100-checkpoint-432"},
    },
    "Olmo-3-7B": {
        "Base":      {"eval": "unlearning-cleanslate/eval-olmo-3-7b",
                      "gen":  "unlearning-cleanslate/generations-olmo-3-7b-pre_val"},
        "33%":       {"eval": "unlearning-cleanslate/eval-checkpoint-64-w100-s10",
                      "gen":  "arpandeepk/generations-olmo-3-1025-7b-simnpo-gentle-checkpoint-64"},
        "66%":       {"eval": "unlearning-cleanslate/eval-checkpoint-128-w100-s10",
                      "gen":  "arpandeepk/generations-olmo-3-1025-7b-simnpo-gentle-checkpoint-128"},
        "100%":      {"eval": "unlearning-cleanslate/eval-checkpoint-192-w100-s10",
                      "gen":  "arpandeepk/generations-olmo-3-1025-7b-simnpo-gentle-checkpoint-192"},
        "Aggressive":{"eval": None,
                      "gen":  "unlearning-cleanslate/generations-olmo-3-7b-simnpo-baseline"},
    },
    "Qwen3-8B": {
        "Base":      {"eval": "unlearning-cleanslate/eval-qwen3-8b",
                      "gen":  "unlearning-cleanslate/generations-qwen3-8b-pre_val"},
        "33%":       {"eval": "unlearning-cleanslate/eval-checkpoint-80-w100-s10",
                      "gen":  "arpandeepk/generations-qwen3-8b-simnpo-gentle-checkpoint-80"},
        "66%":       {"eval": "unlearning-cleanslate/eval-checkpoint-160-w100-s10",
                      "gen":  "arpandeepk/generations-qwen3-8b-simnpo-gentle-checkpoint-160"},
        "100%":      {"eval": "unlearning-cleanslate/eval-checkpoint-240-w100-s10",
                      "gen":  "arpandeepk/generations-qwen3-8b-simnpo-gentle-checkpoint-240"},
        "Aggressive":{"eval": None,
                      "gen":  None},
    },
}

# Use target-100 forget pool (since these checkpoints were trained at target-100).
CKPT_FSID = {
    "Llama-3.1-8B": "unlearning-cleanslate/fsid-curated-llama-8b-target-100",
    "Olmo-3-7B":    "unlearning-cleanslate/fsid-curated-olmo-7b-target-100",
    "Qwen3-8B":     "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100",
}

# Algorithm comparison: SimNPO / UNDIAL / RMU at Oracle baseline (target-50)
ALGO_STAGES = ["SimNPO", "UNDIAL", "RMU"]


def _algo_paths_simnpo(model_short: str) -> dict[str, str]:
    return {
        "Base":     f"unlearning-cleanslate/eval-{model_short}",
        "PreGen":   f"unlearning-cleanslate/generations-{model_short}-pre_val",
        "PostEval": f"unlearning-cleanslate/eval-{model_short}-simnpo-gentle-baseline",
        "PostGen":  f"unlearning-cleanslate/generations-{model_short}-simnpo-gentle-baseline",
    }


def _algo_paths(model_short: str, algo: str) -> dict[str, str]:
    suffix = {"SimNPO": "simnpo-gentle", "UNDIAL": "undial", "RMU": "rmu"}[algo]
    return {
        "Base":     f"unlearning-cleanslate/eval-{model_short}",
        "PreGen":   f"unlearning-cleanslate/generations-{model_short}-pre_val",
        "PostEval": f"unlearning-cleanslate/eval-{model_short}-{suffix}-baseline",
        "PostGen":  f"unlearning-cleanslate/generations-{model_short}-{suffix}-baseline",
    }


# ── Generic 3-panel renderer ──────────────────────────────────────────────────
def _render_3panel(
    series_F: dict[str, list[float | None]],
    series_R: dict[str, list[float | None]],
    series_V: dict[str, list[float | None]],
    x_labels: list[str],
    out: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "axes.linewidth": 2.0,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 13.5,
    })
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=160)
    x = np.arange(len(x_labels))

    panels = [
        (axes[0], series_F, "Forget unmemorized (%)",          "{:.0f}", False),
        (axes[1], series_R, "Retain unmemorized (%)",          "{:.0f}", False),
        (axes[2], series_V, "Mean Δ% on retention",            "{:+.1f}", True),
    ]

    for ax, series, ylabel, fmt, is_delta in panels:
        for model, ys in series.items():
            color = MODEL_COLORS.get(model, "#888888")
            xs_v = [xi for xi, y in zip(x, ys) if y is not None]
            ys_v = [y for y in ys if y is not None]
            if len(xs_v) < 2:
                continue
            ax.plot(xs_v, ys_v, marker="D", color=color,
                    linewidth=4.5, markersize=13, markeredgecolor="white",
                    markeredgewidth=1.8, alpha=0.95, zorder=3, label=model)
            xi_e, yi_e = xs_v[-1], ys_v[-1]
            ax.annotate(fmt.format(yi_e), xy=(xi_e, yi_e), xytext=(12, 0),
                        textcoords="offset points", va="center",
                        fontsize=14, fontweight="bold", color=color)
        # Ideal-state reference: 0 (delta) or 0 (mem floor — never achieved at base)
        ax.axhline(0, color="#E63946", linewidth=3.5, linestyle="--",
                   alpha=0.9, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_xlabel("Stage" if not is_delta else "Stage", labelpad=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.set_xlim(-0.4, len(x) - 0.2)

    axes[0].legend(loc="best", framealpha=0.95, fontsize=12.5,
                   frameon=True, fancybox=False)
    axes[0].get_legend().get_frame().set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    if out.suffix.lower() == ".pdf":
        png = out.with_suffix(".png")
        fig.savefig(png, bbox_inches="tight", dpi=200)
        print(f"saved {out} and {png}")
    else:
        print(f"saved {out}")
    plt.close(fig)


# ── Trajectory plot (checkpoints) ─────────────────────────────────────────────
def plot_checkpoint_trajectory(out: Path = OUT_DIR / "delta_checkpoints.pdf") -> None:
    series_F: dict[str, list[float | None]] = {}
    series_R: dict[str, list[float | None]] = {}
    series_V: dict[str, list[float | None]] = {}

    for model, stages in CKPT_PATHS.items():
        forget_ids = B._cache_forget(CKPT_FSID[model], "baseline")
        if forget_ids is None:
            continue
        # Base eval / gen for the AvgVal Δ baseline
        base_gen = stages["Base"]["gen"]
        base_eval = stages["Base"]["eval"]
        base_split = _mem_split(base_eval, forget_ids) if base_eval else None

        F_arr, R_arr, V_arr = [], [], []
        for stage in CKPT_STAGES:
            entry = stages.get(stage) or {"eval": None, "gen": None}
            # F% / R% (relative to BASE memorization counts on the same content_ids)
            if base_split is None or entry["eval"] is None:
                F_arr.append(None); R_arr.append(None)
            else:
                bF_w, bF_m, bR_w, bR_m = base_split
                cur_split = _mem_split(entry["eval"], forget_ids)
                if cur_split is None:
                    F_arr.append(None); R_arr.append(None)
                else:
                    cF_w, cF_m, cR_w, cR_m = cur_split
                    F_arr.append((bF_m - cF_m) / bF_m * 100 if bF_m else None)
                    R_arr.append((bR_m - cR_m) / bR_m * 100 if bR_m else None)
            # AvgVal Δpp (relative to base gen)
            if base_gen is None or entry["gen"] is None:
                V_arr.append(None)
            else:
                V_arr.append(_retention_avg_delta_pp(base_gen, entry["gen"]))
        series_F[model] = F_arr
        series_R[model] = R_arr
        series_V[model] = V_arr

    _render_3panel(series_F, series_R, series_V, CKPT_STAGES, out)


# ── Algorithm comparison plot ─────────────────────────────────────────────────
def plot_algo_comparison(out: Path = OUT_DIR / "delta_algos.pdf") -> None:
    series_F: dict[str, list[float | None]] = {}
    series_R: dict[str, list[float | None]] = {}
    series_V: dict[str, list[float | None]] = {}

    fsid_short = {"Llama-3.1-8B": "llama-8b", "Olmo-3-7B": "olmo-7b", "Qwen3-8B": "qwen3-8b"}
    base_clean = {"Llama-3.1-8B": "llama-3_1-8b", "Olmo-3-7B": "olmo-3-7b", "Qwen3-8B": "qwen3-8b"}

    for model in MODEL_COLORS:
        forget_ids = B._cache_forget(f"unlearning-cleanslate/fsid-curated-{fsid_short[model]}", "baseline")
        if forget_ids is None:
            continue
        pre_eval = f"unlearning-cleanslate/eval-{base_clean[model]}"
        pre_gen = f"unlearning-cleanslate/generations-{base_clean[model]}-pre_val"
        base_split = _mem_split(pre_eval, forget_ids)

        F_arr, R_arr, V_arr = [], [], []
        for algo in ALGO_STAGES:
            paths = _algo_paths(base_clean[model], algo)
            cur_split = _mem_split(paths["PostEval"], forget_ids) if paths["PostEval"] else None
            if base_split and cur_split:
                bF_w, bF_m, bR_w, bR_m = base_split
                cF_w, cF_m, cR_w, cR_m = cur_split
                F_arr.append((bF_m - cF_m) / bF_m * 100 if bF_m else None)
                R_arr.append((bR_m - cR_m) / bR_m * 100 if bR_m else None)
            else:
                F_arr.append(None); R_arr.append(None)
            V_arr.append(_retention_avg_delta_pp(pre_gen, paths["PostGen"]))
        series_F[model] = F_arr
        series_R[model] = R_arr
        series_V[model] = V_arr

    _render_3panel(series_F, series_R, series_V, ALGO_STAGES, out)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", choices=("ckpt", "algo", "both"), default="both")
    args = ap.parse_args()
    if args.kind in ("ckpt", "both"):
        plot_checkpoint_trajectory()
    if args.kind in ("algo", "both"):
        plot_algo_comparison()
    return 0


if __name__ == "__main__":
    sys.exit(main())
