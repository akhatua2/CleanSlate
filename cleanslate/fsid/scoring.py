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

    # Window-level memorization transitions
    forget_windows_memorized_pre: int | None = None
    forget_windows_memorized_post: int | None = None
    forget_windows_delta_memorized: int | None = None
    retain_windows_memorized_pre: int | None = None
    retain_windows_memorized_post: int | None = None
    retain_windows_delta_memorized: int | None = None

    # Aggregated benchmark groups (BBH, MMLU)
    bbh_pre_eq_weighted: float | None = None
    bbh_post_eq_weighted: float | None = None
    bbh_delta_eq_weighted: float | None = None
    bbh_pre_sample_weighted: float | None = None
    bbh_post_sample_weighted: float | None = None
    bbh_delta_sample_weighted: float | None = None
    bbh_n_subtasks: int | None = None
    bbh_n_samples: int | None = None
    mmlu_pre_eq_weighted: float | None = None
    mmlu_post_eq_weighted: float | None = None
    mmlu_delta_eq_weighted: float | None = None
    mmlu_pre_sample_weighted: float | None = None
    mmlu_post_sample_weighted: float | None = None
    mmlu_delta_sample_weighted: float | None = None
    mmlu_n_subtasks: int | None = None
    mmlu_n_samples: int | None = None

    # GSM8K filter split (both already scored side-by-side)
    gsm8k_strict_pre: float | None = None
    gsm8k_strict_post: float | None = None
    gsm8k_strict_delta: float | None = None
    gsm8k_flexible_pre: float | None = None
    gsm8k_flexible_post: float | None = None
    gsm8k_flexible_delta: float | None = None

    # cleanslate_qa split by content_id in forget_ids
    qa_forget_pre: float | None = None
    qa_forget_post: float | None = None
    qa_forget_delta: float | None = None
    qa_forget_n: int | None = None
    qa_retain_pre: float | None = None
    qa_retain_post: float | None = None
    qa_retain_delta: float | None = None
    qa_retain_n: int | None = None

    # Per-task generation-derived deltas (one row per surface)
    generation_deltas: list[dict[str, Any]] | None = None
    utility_retained: float | None = None
    utility_retained_sample_weighted: float | None = None

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


def _aggregate_generations(
    pre_repo: str,
    post_repo: str,
    *,
    forget_ids: set[str],
) -> dict[str, Any]:
    """Compute per-task pre/post/delta from pushed generations datasets.

    For every config that exists in both pre_repo and post_repo:
      - mean of `score` column = canonical task accuracy
      - if task == "gsm8k": split rows by `filter` field, report strict-match
        and flexible-extract separately
      - if task == "cleanslate_qa": split rows by doc.content_id in forget_ids
    Then merge bbh_* (27 subtasks) and mmlu_* (57 subtasks) into single
    equal-weighted and sample-weighted aggregates.

    Fails loudly on schema or coverage problems.
    """
    from datasets import get_dataset_config_names, load_dataset

    pre_cfgs = set(get_dataset_config_names(pre_repo))
    post_cfgs = set(get_dataset_config_names(post_repo))
    only_pre = pre_cfgs - post_cfgs
    only_post = post_cfgs - pre_cfgs
    if only_pre or only_post:
        raise ValueError(
            f"config mismatch between {pre_repo} and {post_repo}: "
            f"only-in-pre={sorted(only_pre)}, only-in-post={sorted(only_post)}"
        )
    configs = sorted(pre_cfgs)

    def _scores(repo: str, cfg: str) -> list[dict[str, Any]]:
        ds = load_dataset(repo, cfg, split="train")
        if "score" not in ds.column_names:
            raise KeyError(f"generations dataset {repo}/{cfg} has no 'score' column")
        return list(ds)

    def _row(name: str, pre: float, post: float, n: int) -> dict[str, Any]:
        return {
            "benchmark": name,
            "pre": pre,
            "post": post,
            "delta": post - pre,
            "retention": (post / pre) if pre > 0 else 0.0,
            "n": n,
        }

    bbh_subs: list[tuple[str, float, float, int]] = []
    mmlu_subs: list[tuple[str, float, float, int]] = []
    standalone_rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {}

    for cfg in configs:
        pre_rows = _scores(pre_repo, cfg)
        post_rows = _scores(post_repo, cfg)
        pre_acc = _mean([float(r["score"] or 0.0) for r in pre_rows])
        post_acc = _mean([float(r["score"] or 0.0) for r in post_rows])
        n = len(pre_rows)

        if cfg.startswith("bbh_cot_fewshot_"):
            bbh_subs.append((cfg, pre_acc, post_acc, n))
            continue
        if cfg.startswith("mmlu_"):
            mmlu_subs.append((cfg, pre_acc, post_acc, n))
            continue
        if cfg == "gsm8k":
            for filt in ("strict-match", "flexible-extract"):
                pre_f = [float(r["score"] or 0.0) for r in pre_rows if r.get("filter") == filt]
                post_f = [float(r["score"] or 0.0) for r in post_rows if r.get("filter") == filt]
                if not pre_f or not post_f:
                    raise ValueError(
                        f"gsm8k generations missing '{filt}' filter rows "
                        f"(pre n={len(pre_f)}, post n={len(post_f)}) — re-generate with both filters"
                    )
                p_acc = _mean(pre_f)
                q_acc = _mean(post_f)
                standalone_rows.append(_row(f"gsm8k_{filt}", p_acc, q_acc, len(pre_f)))
                out[f"gsm8k_{filt.replace('-', '_').replace('flexible_extract', 'flexible').replace('strict_match', 'strict')}_pre"] = p_acc
                out[f"gsm8k_{filt.replace('-', '_').replace('flexible_extract', 'flexible').replace('strict_match', 'strict')}_post"] = q_acc
                out[f"gsm8k_{filt.replace('-', '_').replace('flexible_extract', 'flexible').replace('strict_match', 'strict')}_delta"] = q_acc - p_acc
            continue
        if cfg == "cleanslate_qa":
            standalone_rows.append(_row(cfg, pre_acc, post_acc, n))

            def _doc_cid(r: dict[str, Any]) -> str:
                doc = r.get("doc") or {}
                cid = doc.get("content_id")
                if cid is None:
                    raise KeyError("cleanslate_qa rows lack doc.content_id; cannot split forget/retain")
                return str(cid)

            f_pre = [float(r["score"] or 0.0) for r in pre_rows if _doc_cid(r) in forget_ids]
            r_pre = [float(r["score"] or 0.0) for r in pre_rows if _doc_cid(r) not in forget_ids]
            f_post = [float(r["score"] or 0.0) for r in post_rows if _doc_cid(r) in forget_ids]
            r_post = [float(r["score"] or 0.0) for r in post_rows if _doc_cid(r) not in forget_ids]
            qa_f_pre = _mean(f_pre); qa_f_post = _mean(f_post)
            qa_r_pre = _mean(r_pre); qa_r_post = _mean(r_post)
            standalone_rows.append(_row("cleanslate_qa-forget", qa_f_pre, qa_f_post, len(f_pre)))
            standalone_rows.append(_row("cleanslate_qa-retain", qa_r_pre, qa_r_post, len(r_pre)))
            out["qa_forget_pre"] = qa_f_pre
            out["qa_forget_post"] = qa_f_post
            out["qa_forget_delta"] = qa_f_post - qa_f_pre
            out["qa_forget_n"] = len(f_pre)
            out["qa_retain_pre"] = qa_r_pre
            out["qa_retain_post"] = qa_r_post
            out["qa_retain_delta"] = qa_r_post - qa_r_pre
            out["qa_retain_n"] = len(r_pre)
            continue

        standalone_rows.append(_row(cfg, pre_acc, post_acc, n))

    def _aggregate_group(label: str, subs: list[tuple[str, float, float, int]]) -> None:
        if not subs:
            return
        eq_pre = _mean([s[1] for s in subs])
        eq_post = _mean([s[2] for s in subs])
        total_n = sum(s[3] for s in subs)
        sw_pre = sum(s[1] * s[3] for s in subs) / total_n if total_n else 0.0
        sw_post = sum(s[2] * s[3] for s in subs) / total_n if total_n else 0.0
        out[f"{label}_pre_eq_weighted"] = eq_pre
        out[f"{label}_post_eq_weighted"] = eq_post
        out[f"{label}_delta_eq_weighted"] = eq_post - eq_pre
        out[f"{label}_pre_sample_weighted"] = sw_pre
        out[f"{label}_post_sample_weighted"] = sw_post
        out[f"{label}_delta_sample_weighted"] = sw_post - sw_pre
        out[f"{label}_n_subtasks"] = len(subs)
        out[f"{label}_n_samples"] = total_n
        # Use equal-weighted for the merged generation_deltas row (paper convention).
        standalone_rows.append({
            "benchmark": label,
            "pre": eq_pre,
            "post": eq_post,
            "delta": eq_post - eq_pre,
            "retention": (eq_post / eq_pre) if eq_pre > 0 else 0.0,
            "n": total_n,
            "n_subtasks": len(subs),
        })

    _aggregate_group("bbh", bbh_subs)
    _aggregate_group("mmlu", mmlu_subs)

    standalone_rows.sort(key=lambda r: r["benchmark"])
    out["generation_deltas"] = standalone_rows

    # Utility retained over the merged surface (each row counts once):
    # exclude the cleanslate_qa-forget/retain split rows from the average so
    # they don't double-count the cleanslate_qa parent.
    util_rows = [r for r in standalone_rows if r["benchmark"] not in ("cleanslate_qa-forget", "cleanslate_qa-retain")]
    if util_rows:
        out["utility_retained"] = _mean([r["retention"] for r in util_rows])
        total_n = sum(r["n"] for r in util_rows) or 1
        out["utility_retained_sample_weighted"] = sum(r["retention"] * r["n"] for r in util_rows) / total_n
    return out


def compute_score(
    *,
    forget_set_path: str | Path,
    pre_eval_path: str | Path,
    post_eval_path: str | Path,
    pre_generations_path: str,
    post_generations_path: str,
) -> UnlearningScore:
    """Compute a benchmark score from local pre/post eval artifacts.

    If pre/post validation JSONs are supplied, their `ValidationDelta`s are
    serialized onto the result so the single score.json carries both forget
    and retention numbers (what the paper's table row consumes).
    """
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
        f_post_mem = f_ret_mem + f_reg
        window_agg["forget_total_windows"] = f_total
        window_agg["forget_windows_forgotten"] = f_forg
        window_agg["forget_windows_regressed"] = f_reg
        window_agg["forget_window_forget_rate"] = f_forg / f_pre_mem if f_pre_mem > 0 else 0.0
        window_agg["forget_windows_memorized_pre"] = f_pre_mem
        window_agg["forget_windows_memorized_post"] = f_post_mem
        window_agg["forget_windows_delta_memorized"] = f_pre_mem - f_post_mem
    if any(d.num_windows is not None for d in retain_deltas):
        r_total = sum(d.num_windows for d in retain_deltas if d.num_windows is not None)
        r_reg = sum(d.windows_regressed for d in retain_deltas if d.windows_regressed is not None)
        r_forg = sum(d.windows_forgotten for d in retain_deltas if d.windows_forgotten is not None)
        r_ret_mem = sum(
            d.windows_retained_memorized for d in retain_deltas if d.windows_retained_memorized is not None
        )
        r_pre_mem = r_forg + r_ret_mem
        r_post_mem = r_ret_mem + r_reg
        window_agg["retain_total_windows"] = r_total
        window_agg["retain_windows_regressed"] = r_reg
        window_agg["retain_window_stability"] = 1.0 - (r_reg / r_total) if r_total > 0 else 1.0
        window_agg["retain_windows_memorized_pre"] = r_pre_mem
        window_agg["retain_windows_memorized_post"] = r_post_mem
        window_agg["retain_windows_delta_memorized"] = r_pre_mem - r_post_mem

    generation_kwargs = _aggregate_generations(
        str(pre_generations_path),
        str(post_generations_path),
        forget_ids=forget_ids,
    )

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
        **generation_kwargs,
    )


def save_score(score: UnlearningScore, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score.to_dict(), indent=2))
    return out
