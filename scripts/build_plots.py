"""
Plots for the curation paper, sourced from the same HF cache as
build_tables_from_hf.py.

Renders three figures into results/plots/ :
  pareto.pdf       F vs AvgVal Δ (the curation trade-off)
  heatmap.pdf      per-benchmark Δ heatmap across all (model × curation)
  calibration.pdf  R% vs AvgVal Δ (does retain memorization predict capability?)

Run after `build_tables_from_hf.py --warm` so the cache is populated.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the table script's data layer so we don't re-implement HF / score logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tables_from_hf as B  # noqa: E402

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data assembly ─────────────────────────────────────────────────────────────
def _delta_pctchg(row: dict | None) -> float | None:
    if not row:
        return None
    pre, post = row.get("pre"), row.get("post")
    if pre is None or post is None or pre == 0:
        return None
    return (post - pre) / pre * 100


def _F_pct(score: dict) -> float | None:
    f = score.get("forget", {})
    if not f.get("pre"):
        return None
    return (f["pre"] - f["post"]) / f["pre"] * 100


def _R_pct(score: dict) -> float | None:
    r = score.get("retain", {})
    if not r.get("pre"):
        return None
    return (r["pre"] - r["post"]) / r["pre"] * 100


def _avg_val(score: dict) -> float | None:
    ret = score.get("retention", {})
    deltas = []
    for _, key in B.VAL_COLS:
        d = _delta_pctchg(ret.get(key))
        if d is not None:
            deltas.append(d)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _qa(score: dict) -> float | None:
    return _delta_pctchg(score.get("retention", {}).get("cleanslate_qa"))


def collect_points() -> list[dict]:
    """One record per (model, curation_label, algo_label, score) cell.
    Includes retrieval (BM25 mid / IGM mid / BM25 pre) + Oracle for SimNPO,
    and the algorithm-comparison cells (Oracle × SimNPO/UNDIAL/RMU)."""
    out: list[dict] = []

    # Retrieval cells: 3 rules × 6 models, all SimNPO-gentle
    for label, rule_split, rule_slug in B.RULES_RETRIEVAL:
        for m in B.MODELS_RETRIEVAL:
            score = B._build_score(m, rule_split, rule_slug, oracle=False)
            if not score:
                continue
            out.append({
                "model": m.display, "fsid_short": m.fsid_short,
                "curation": label, "algo": "SimNPO",
                "score": score,
            })

    # Oracle cells: SimNPO at baseline
    rule_split, rule_slug = B.RULE_ORACLE
    for m in B.MODELS_ORACLE:
        score = B._build_score(m, rule_split, rule_slug, oracle=True)
        if not score:
            continue
        out.append({
            "model": m.display, "fsid_short": m.fsid_short,
            "curation": "Oracle", "algo": "SimNPO",
            "score": score,
        })

    # Algo comparison: Oracle × {SimNPO, UNDIAL, RMU} on 3 small models
    for algo_label, suffix in B.ALGOS:
        if algo_label == "SimNPO":
            continue  # already covered above for these models via Oracle row
        for m in B.MODELS_ALGOS:
            paths = B._paths_for_algo(m, suffix)
            forget_ids = B._cache_forget(paths["fsid"], "baseline")
            if forget_ids is None:
                continue
            pre_per = B._cache_eval(paths["pre_eval"])
            post_per = B._cache_eval(paths["post_eval"])
            if pre_per is None or post_per is None:
                continue
            all_ids = set(pre_per.keys())
            retain_ids = all_ids - forget_ids
            pre_f = B._mem_block(pre_per, forget_ids); post_f = B._mem_block(post_per, forget_ids)
            pre_r = B._mem_block(pre_per, retain_ids); post_r = B._mem_block(post_per, retain_ids)
            if not all([pre_f, post_f, pre_r, post_r]):
                continue
            retention = B._retention_from_cache(paths["pre_gen"], paths["post_gen"], retain_ids)
            score = {
                "forget": {"pre": pre_f[1], "post": post_f[1]},
                "retain": {"pre": pre_r[1], "post": post_r[1]},
                "retention": retention,
            }
            out.append({
                "model": m.display, "fsid_short": m.fsid_short,
                "curation": "Oracle", "algo": algo_label,
                "score": score,
            })

    return out


# ── Plot 1: Curation trade-off (paired panels) ────────────────────────────────
# One pastel colour per MODEL (matching the reference figure's per-model lines).
_MODEL_COLORS = {
    "Llama-3.1-8B": "#7AA9D6",   # dusty blue
    "Olmo-3-7B":    "#C7A8DA",   # lavender
    "Qwen3-8B":     "#F4A992",   # peach
    "Nemotron-9B":  "#7AC1A8",   # mint
    "Gemma-3-12B":  "#8AC57A",   # sage green
    "Olmo-3-32B":   "#E0C97A",   # gold
}
_CURATION_ORDER = ["Oracle", "BM25 mid", "Infinigram mid", "BM25 pre"]
_CURATION_SHORT = ["Oracle", "BM25\nmid", "IGM\nmid", "BM25\npre"]


_CURATION_COLORS = {
    "Oracle":         "#E07B6B",  # coral red
    "BM25 mid":       "#7AA9D6",  # dusty blue
    "Infinigram mid": "#8AC57A",  # sage green
    "BM25 pre":       "#C7A8DA",  # lavender
}


def plot_pareto(points: list[dict], path: Path) -> None:
    """Per-curation 2D mean ± std on (F%, capability Δ%) — one ellipse per
    curation showing where each method lands and how variable it is."""
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
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 14,
        "legend.title_fontsize": 15,
    })
    import numpy as np

    # Only the 4 models we have full retrieval-table data for.
    INCLUDED = {"Llama-3.1-8B", "Olmo-3-7B", "Qwen3-8B", "Gemma-3-12B"}
    pts = [p for p in points if p["algo"] == "SimNPO" and p["model"] in INCLUDED]

    # Per-(model, curation) values for F, R, AvgVal
    metrics = {
        "f": {"label": "Forget unmemorized (%)",     "fn": _F_pct,    "fmt": "{:.0f}"},
        "r": {"label": "Retain unmemorized (%)",     "fn": _R_pct,    "fmt": "{:.0f}"},
        "v": {"label": "Mean Δ% on retention",        "fn": _avg_val, "fmt": "{:+.1f}"},
    }
    per_model: dict[str, dict[str, list[float | None]]] = {}
    for p in pts:
        cur = p["curation"]
        if cur not in _CURATION_ORDER:
            continue
        idx = _CURATION_ORDER.index(cur)
        m = p["model"]
        per_model.setdefault(m, {k: [None] * len(_CURATION_ORDER) for k in metrics})
        for k, info in metrics.items():
            v = info["fn"](p["score"])
            if v is not None:
                per_model[m][k][idx] = v

    # Per-model colours (pastel palette like the reference figure)
    palette = [
        "#7AA9D6",   # dusty blue
        "#C7A8DA",   # lavender
        "#F4A992",   # peach
        "#8AC57A",   # sage green
        "#7AC1A8",   # mint
        "#E0C97A",   # gold
    ]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(per_model)}

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=160)
    x = np.arange(len(_CURATION_ORDER))

    for ax, (k, info) in zip(axes, metrics.items()):
        for m, series in per_model.items():
            ys = series[k]
            xs_valid = [xi for xi, y in zip(x, ys) if y is not None]
            ys_valid = [y for y in ys if y is not None]
            if len(xs_valid) < 2:
                continue
            ax.plot(xs_valid, ys_valid, marker="D", color=colors[m],
                    linewidth=4.5, markersize=13, markeredgecolor="white",
                    markeredgewidth=1.8, alpha=0.95, zorder=3, label=m)
            # End-of-line annotation (rightmost valid point)
            xi_end, yi_end = xs_valid[-1], ys_valid[-1]
            ax.annotate(info["fmt"].format(yi_end),
                        xy=(xi_end, yi_end), xytext=(12, 0),
                        textcoords="offset points", va="center",
                        fontsize=14, fontweight="bold", color=colors[m])
        # Ideal-state reference line (zero)
        ax.axhline(0, color="#E63946", linewidth=3.5, linestyle="--",
                   alpha=0.9, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(_CURATION_SHORT)
        ax.set_ylabel(info["label"], labelpad=10)
        ax.set_xlabel("Curation method", labelpad=10)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.4)
        # Pad right to accommodate end-labels
        ax.set_xlim(-0.4, len(x) - 0.2)

    # Legend goes only on the leftmost panel (cleaner)
    axes[0].legend(loc="best", framealpha=0.95, fontsize=11.5, frameon=True,
                   fancybox=False)
    axes[0].get_legend().get_frame().set_linewidth(1.0)

    return _finish(fig, path)


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=220)
        print(f"saved {path} and {png_path}")
    else:
        print(f"saved {path}")
    plt.close(fig)


# ── Plot 2: per-benchmark damage heatmap ──────────────────────────────────────
def plot_heatmap(points: list[dict], path: Path) -> None:
    bench_keys = [k for _, k in B.VAL_COLS]  # 7 keys
    bench_labels = [n for n, _ in B.VAL_COLS]

    rows: list[tuple[str, list[float]]] = []
    # Order: SimNPO retrieval (BM25mid, IGMmid, BM25pre) per model, then Oracle SimNPO,
    # then Oracle UNDIAL/RMU
    order = [
        ("SimNPO", "BM25 mid"),
        ("SimNPO", "Infinigram mid"),
        ("SimNPO", "BM25 pre"),
        ("SimNPO", "Oracle"),
        ("UNDIAL", "Oracle"),
        ("RMU",    "Oracle"),
    ]
    for algo, cur in order:
        for m in (*B.MODELS_RETRIEVAL,):
            for p in points:
                if p["algo"] != algo or p["curation"] != cur or p["fsid_short"] != m.fsid_short:
                    continue
                ret = p["score"].get("retention", {})
                row = []
                for k in bench_keys:
                    d = _delta_pctchg(ret.get(k))
                    row.append(np.nan if d is None else d)
                label = f"{algo} / {cur} – {m.display}"
                rows.append((label, row))
                break

    if not rows:
        print("heatmap: no rows to plot")
        return
    labels, data = zip(*rows)
    arr = np.array(data, dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(rows))))
    vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else 30.0
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(bench_labels))); ax.set_xticklabels(bench_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                        color="black" if abs(v) < vmax * 0.5 else "white",
                        fontsize=7)
    ax.set_title("Δ% per benchmark (rows = (algo / curation − model))")
    fig.colorbar(im, ax=ax, label="Δ% (relative to baseline)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


# ── Plot 3: R vs AvgVal calibration ───────────────────────────────────────────
def plot_calibration(points: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, ys, colors, markers = [], [], [], []
    for p in points:
        if p["algo"] != "SimNPO":
            continue
        r = _R_pct(p["score"]); v = _avg_val(p["score"])
        if r is None or v is None:
            continue
        xs.append(r); ys.append(v)
        colors.append(_CURATION_COLORS.get(p["curation"], "grey"))
        markers.append(_MODEL_MARKERS.get(p["model"], "o"))

    for x, y, c, mk in zip(xs, ys, colors, markers):
        ax.scatter(x, y, c=c, marker=mk, s=110, edgecolor="black",
                   linewidth=0.6, alpha=0.85)

    if len(xs) >= 3:
        coef = np.polyfit(xs, ys, 1)
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, np.polyval(coef, xline), color="grey",
                linewidth=1, linestyle="--",
                label=f"linear fit: y={coef[0]:+.2f}·x{coef[1]:+.1f}")
        # Pearson r
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.02, 0.96, f"Pearson r = {r:+.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Retain unmemorized (%) — higher means more retain-pool damage")
    ax.set_ylabel("Mean Δ% on retention benchmarks")
    ax.set_title("Retain-memorization vs capability loss (does R predict AvgVal Δ?)")
    cur_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=10, label=lbl)
        for lbl, c in _CURATION_COLORS.items()
    ]
    ax.legend(handles=cur_handles, title="Curation", loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {path}")


def main() -> int:
    points = collect_points()
    if not points:
        print("no points — run `build_tables_from_hf.py --warm` first", file=sys.stderr)
        return 1
    print(f"collected {len(points)} points")
    plot_pareto(points, OUT_DIR / "pareto.pdf")
    plot_heatmap(points, OUT_DIR / "heatmap.pdf")
    plot_calibration(points, OUT_DIR / "calibration.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
