"""Post-unlearning scoring: compare pre/post memorization artifacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cleanslate.fsid.datasets.io import load_local_records


@dataclass
class WindowDelta:
    """Per-window change in memorization after unlearning."""

    idx: int
    pre_p_z: float
    post_p_z: float
    delta_p_z: float  # pre - post  (positive = less memorized)
    pre_is_memorized: bool
    post_is_memorized: bool
    transition: str  # "forgotten" | "retained_memorized" | "retained_clean" | "regressed"


@dataclass
class ContentDelta:
    """Per-content change in memorization after unlearning."""

    content_id: str
    content_title: str
    pre_memorized_fraction: float
    post_memorized_fraction: float
    delta_memorized_fraction: float
    pre_max_p_z: float
    post_max_p_z: float
    delta_max_p_z: float
    in_forget_set: bool
    pre_mean_perplexity: float | None = None
    post_mean_perplexity: float | None = None
    pre_min_perplexity: float | None = None
    post_min_perplexity: float | None = None
    num_windows: int | None = None
    windows_forgotten: int | None = None
    windows_retained_memorized: int | None = None
    windows_retained_clean: int | None = None
    windows_regressed: int | None = None
    window_deltas: list[WindowDelta] | None = None


@dataclass
class UnlearningScore:
    """Aggregate unlearning benchmark score."""

    forget_count: int
    forget_mean_delta_frac: float
    forget_mean_delta_pz: float
    forget_fraction_improved: float
    retain_count: int
    retain_mean_abs_delta_frac: float
    retain_mean_abs_delta_pz: float
    forget_score: float
    retain_penalty: float
    per_content: list[ContentDelta]

    forget_total_windows: int | None = None
    forget_windows_forgotten: int | None = None
    forget_windows_regressed: int | None = None
    forget_window_forget_rate: float | None = None
    retain_total_windows: int | None = None
    retain_windows_regressed: int | None = None
    retain_window_stability: float | None = None

    forget_mean_delta_mean_ppl: float | None = None
    forget_mean_delta_min_ppl: float | None = None
    retain_mean_abs_delta_mean_ppl: float | None = None
    retain_mean_abs_delta_min_ppl: float | None = None

    validation_deltas: list[dict[str, Any]] | None = None
    utility_retained: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for item in d.get("per_content", []):
            item.pop("window_deltas", None)
        return d


def _require(row: dict[str, Any], key: str, label: str) -> Any:
    value = row.get(key)
    if value is None:
        raise KeyError(f"{label} row content_id={row.get('content_id')!r} missing {key!r}")
    return value


def _classify_transition(pre_mem: bool, post_mem: bool) -> str:
    match (pre_mem, post_mem):
        case (True, False):
            return "forgotten"
        case (True, True):
            return "retained_memorized"
        case (False, False):
            return "retained_clean"
        case _:
            return "regressed"


def _compute_window_deltas(
    pre_windows: list[dict[str, Any]] | None,
    post_windows: list[dict[str, Any]] | None,
) -> tuple[int, int, int, int, int, list[WindowDelta]] | None:
    """Match pre/post windows by idx and classify transitions."""
    if not pre_windows or not post_windows:
        return None
    pre_by_idx = {int(w["idx"]): w for w in pre_windows}
    post_by_idx = {int(w["idx"]): w for w in post_windows}
    common = sorted(set(pre_by_idx) & set(post_by_idx))
    if not common:
        return None

    counts = {"forgotten": 0, "retained_memorized": 0, "retained_clean": 0, "regressed": 0}
    deltas: list[WindowDelta] = []
    for idx in common:
        pre = pre_by_idx[idx]
        post = post_by_idx[idx]
        pre_pz = float(pre.get("p_z", 0.0))
        post_pz = float(post.get("p_z", 0.0))
        pre_mem = bool(pre.get("is_memorized", False))
        post_mem = bool(post.get("is_memorized", False))
        transition = _classify_transition(pre_mem, post_mem)
        counts[transition] += 1
        deltas.append(
            WindowDelta(
                idx=idx,
                pre_p_z=pre_pz,
                post_p_z=post_pz,
                delta_p_z=pre_pz - post_pz,
                pre_is_memorized=pre_mem,
                post_is_memorized=post_mem,
                transition=transition,
            )
        )
    return (
        len(common),
        counts["forgotten"],
        counts["retained_memorized"],
        counts["retained_clean"],
        counts["regressed"],
        deltas,
    )


def _index_unique_by_content_id(
    records: list[dict[str, Any]], *, label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    missing = 0
    for row in records:
        cid = str(row.get("content_id") or "").strip()
        if not cid:
            missing += 1
            continue
        if cid in indexed:
            duplicates.add(cid)
            continue
        indexed[cid] = row
    if missing:
        raise ValueError(f"{label} contains {missing} row(s) without content_id")
    if duplicates:
        raise ValueError(
            f"{label} has duplicate content_id values: {sorted(duplicates)[:5]}. "
            "Evaluation universes must be unique by content_id."
        )
    return indexed


def _extract_forget_ids(records: list[dict[str, Any]]) -> set[str]:
    """Forget sets may have multiple rows per content_id (e.g. corpus_sweep windows)."""
    forget_ids: set[str] = set()
    missing = 0
    for row in records:
        cid = str(row.get("content_id") or "").strip()
        if not cid:
            missing += 1
            continue
        forget_ids.add(cid)
    if missing:
        raise ValueError(f"forget set contains {missing} row(s) without content_id")
    return forget_ids


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_score(
    *,
    forget_set_path: str | Path,
    pre_eval_path: str | Path,
    post_eval_path: str | Path,
) -> UnlearningScore:
    """Compute a benchmark score from local pre/post eval artifacts."""
    forget_ids = _extract_forget_ids(load_local_records(forget_set_path))
    pre_by_id = _index_unique_by_content_id(load_local_records(pre_eval_path), label="pre-eval")
    post_by_id = _index_unique_by_content_id(load_local_records(post_eval_path), label="post-eval")

    shared = set(pre_by_id) & set(post_by_id)
    if not shared:
        raise ValueError("pre-eval and post-eval share no content IDs")
    pre_by_id = {k: v for k, v in pre_by_id.items() if k in shared}
    post_by_id = {k: v for k, v in post_by_id.items() if k in shared}

    missing_forget = forget_ids - shared
    if missing_forget:
        raise ValueError(
            f"{len(missing_forget)} forget-set content(s) missing from eval universe: "
            f"{sorted(missing_forget)[:5]}"
        )

    deltas: list[ContentDelta] = []
    for content_id in sorted(shared):
        pre = pre_by_id[content_id]
        post = post_by_id[content_id]
        pre_frac = float(_require(pre, "memorized_fraction", "Pre-eval"))
        post_frac = float(_require(post, "memorized_fraction", "Post-eval"))
        pre_pz = float(_require(pre, "max_p_z", "Pre-eval"))
        post_pz = float(_require(post, "max_p_z", "Post-eval"))

        ppl_kwargs: dict[str, Any] = {}
        for key in ("mean_perplexity", "min_perplexity"):
            for side, row in (("pre", pre), ("post", post)):
                if (v := row.get(key)) is not None:
                    ppl_kwargs[f"{side}_{key}"] = float(v)

        window_kwargs: dict[str, Any] = {}
        if (wr := _compute_window_deltas(pre.get("windows"), post.get("windows"))) is not None:
            n_win, w_forg, w_ret_mem, w_ret_clean, w_reg, w_deltas = wr
            window_kwargs = {
                "num_windows": n_win,
                "windows_forgotten": w_forg,
                "windows_retained_memorized": w_ret_mem,
                "windows_retained_clean": w_ret_clean,
                "windows_regressed": w_reg,
                "window_deltas": w_deltas,
            }

        deltas.append(
            ContentDelta(
                content_id=content_id,
                content_title=str(pre.get("content_title", "")),
                pre_memorized_fraction=pre_frac,
                post_memorized_fraction=post_frac,
                delta_memorized_fraction=pre_frac - post_frac,
                pre_max_p_z=pre_pz,
                post_max_p_z=post_pz,
                delta_max_p_z=pre_pz - post_pz,
                in_forget_set=(content_id in forget_ids),
                **ppl_kwargs,
                **window_kwargs,
            )
        )

    forget_deltas = [d for d in deltas if d.in_forget_set]
    retain_deltas = [d for d in deltas if not d.in_forget_set]

    forget_fraction_improved = (
        sum(1 for d in forget_deltas if d.delta_memorized_fraction > 0) / len(forget_deltas)
        if forget_deltas else 0.0
    )
    retain_mean_abs_delta_frac = _mean([abs(d.delta_memorized_fraction) for d in retain_deltas])
    pre_retain_mean = _mean([d.pre_memorized_fraction for d in retain_deltas])
    retain_penalty = min(
        retain_mean_abs_delta_frac / pre_retain_mean if pre_retain_mean > 0 else retain_mean_abs_delta_frac,
        1.0,
    )

    ppl_agg: dict[str, Any] = {}
    for side_name, source in (("forget", forget_deltas), ("retain", retain_deltas)):
        for stat in ("mean_perplexity", "min_perplexity"):
            deltas_ppl = [
                getattr(d, f"post_{stat}") - getattr(d, f"pre_{stat}")
                for d in source
                if getattr(d, f"pre_{stat}") is not None and getattr(d, f"post_{stat}") is not None
            ]
            if not deltas_ppl:
                continue
            if side_name == "forget":
                ppl_agg[f"forget_mean_delta_{stat.split('_')[0]}_ppl"] = _mean(deltas_ppl)
            else:
                ppl_agg[f"retain_mean_abs_delta_{stat.split('_')[0]}_ppl"] = _mean([abs(v) for v in deltas_ppl])

    window_agg: dict[str, Any] = {}
    if any(d.num_windows is not None for d in forget_deltas):
        f_total = sum(d.num_windows for d in forget_deltas if d.num_windows is not None)
        f_forg = sum(d.windows_forgotten for d in forget_deltas if d.windows_forgotten is not None)
        f_reg = sum(d.windows_regressed for d in forget_deltas if d.windows_regressed is not None)
        f_ret_mem = sum(
            d.windows_retained_memorized for d in forget_deltas if d.windows_retained_memorized is not None
        )
        f_pre_mem = f_forg + f_ret_mem
        window_agg["forget_total_windows"] = f_total
        window_agg["forget_windows_forgotten"] = f_forg
        window_agg["forget_windows_regressed"] = f_reg
        window_agg["forget_window_forget_rate"] = f_forg / f_pre_mem if f_pre_mem > 0 else 0.0
    if any(d.num_windows is not None for d in retain_deltas):
        r_total = sum(d.num_windows for d in retain_deltas if d.num_windows is not None)
        r_reg = sum(d.windows_regressed for d in retain_deltas if d.windows_regressed is not None)
        window_agg["retain_total_windows"] = r_total
        window_agg["retain_windows_regressed"] = r_reg
        window_agg["retain_window_stability"] = 1.0 - (r_reg / r_total) if r_total > 0 else 1.0

    return UnlearningScore(
        forget_count=len(forget_deltas),
        forget_mean_delta_frac=_mean([d.delta_memorized_fraction for d in forget_deltas]),
        forget_mean_delta_pz=_mean([d.delta_max_p_z for d in forget_deltas]),
        forget_fraction_improved=forget_fraction_improved,
        retain_count=len(retain_deltas),
        retain_mean_abs_delta_frac=retain_mean_abs_delta_frac,
        retain_mean_abs_delta_pz=_mean([abs(d.delta_max_p_z) for d in retain_deltas]),
        forget_score=forget_fraction_improved,
        retain_penalty=retain_penalty,
        per_content=deltas,
        **ppl_agg,
        **window_agg,
    )


def save_score(score: UnlearningScore, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score.to_dict(), indent=2))
    return out
