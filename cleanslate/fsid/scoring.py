"""Post-unlearning scoring: compare pre/post memorization + retention artifacts.

Three entry points, all sharing the same per-side primitives:

  compute_score       -- two-sided diff   (forget_pool, pre_eval, post_eval, pre_gen, post_gen)
  compute_score_pre   -- pre-side only    (forget_pool, pre_eval,  pre_gen)
  compute_score_post  -- post-side only   (forget_pool, post_eval, post_gen, checkpoint)

Output is a nested dict (saved as JSON):

  forget    -- window-level memorization aggregates over forget content_ids
  retain    -- same aggregates over retain content_ids (eval universe minus forget)
  retention -- per-task lm-eval accuracies. Two-sided: {pre, post, delta, n}.
               One-sided: {acc, n}. BBH/MMLU folded equal-weighted; GSM8K
               split by filter; cleanslate_qa carries a forget/retain sub-split.
  meta      -- echo of input paths
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from datasets import get_dataset_config_names, load_dataset


# ---------------------------------------------------------------------------
# Loaders + sanity checks
# ---------------------------------------------------------------------------


def _looks_like_hf_id(s: str) -> bool:
    if s.startswith((".", "/", "~")):
        return False
    base = s.split(":")[0]
    parts = base.split("/")
    return len(parts) == 2 and not Path(base).exists()


def _parse_hf_spec(s: str) -> tuple[str, str | None, str]:
    """`org/repo[:config[:split]]` -> (repo, config_or_None, split)."""
    parts = s.split(":")
    if len(parts) == 1:
        return parts[0], None, "train"
    if len(parts) == 2:
        return parts[0], parts[1], "train"
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"Expected repo[:config[:split]], got {s!r}")


def _load_polars(spec: str | Path) -> pl.DataFrame:
    """Load a parquet/HF-dataset spec into a polars DataFrame."""
    s = str(spec)
    if _looks_like_hf_id(s):
        repo, cfg, split = _parse_hf_spec(s)
        ds = load_dataset(repo, cfg, split=split)
        return ds.to_polars()
    p = Path(s)
    if not p.exists():
        raise FileNotFoundError(f"No local file at {p}; pass a HF dataset id instead.")
    if p.is_dir():
        candidates = sorted(p.glob("**/*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"{p} contains no parquet files")
        return pl.read_parquet([str(c) for c in candidates])
    if p.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported input format {p.suffix!r} for {p}")
    return pl.read_parquet(str(p))


def _require_columns(df: pl.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns {missing}; have {df.columns}")


def _content_id_set(df: pl.DataFrame, label: str) -> set[str]:
    _require_columns(df, ["content_id"], label)
    if df["content_id"].null_count():
        raise ValueError(f"{label} has null content_id rows")
    return set(df["content_id"].to_list())


def _config_to_polars(repo: str, cfg: str) -> pl.DataFrame:
    return load_dataset(repo, cfg, split="train").to_polars()


def _check_scores(df: pl.DataFrame, repo: str, cfg: str) -> None:
    _require_columns(df, ["score"], f"{repo}/{cfg}")
    n_null = df["score"].null_count()
    if n_null:
        raise ValueError(f"{repo}/{cfg} has {n_null} null `score` rows — re-run the judge")


# ---------------------------------------------------------------------------
# Per-side primitives — used by both two-sided and one-sided paths
# ---------------------------------------------------------------------------


def _side_mem(universe: set[str], df: pl.DataFrame, label: str) -> dict:
    """One side's window aggregates over `universe`.

    Returns {n_windows_total, memorized_windows, mean_memorized_pz}. Raises on
    null content_id / is_memorized / p_z, or on universe ⊄ df.content_id.
    """
    _require_columns(df, ["content_id", "windows"], label)
    if not universe:
        raise ValueError(f"{label}: empty universe")

    universe_list = list(universe)
    f = df.filter(pl.col("content_id").is_in(universe_list)).select(["content_id", "windows"])
    missing = universe - set(f["content_id"].to_list())
    if missing:
        raise ValueError(
            f"{len(missing)} content_id(s) missing from {label}. "
            f"missing[:5]={sorted(missing)[:5]}"
        )

    w = f.explode("windows").unnest("windows")
    _require_columns(w, ["is_memorized", "p_z"], f"{label} windows")
    if w["is_memorized"].null_count() or w["p_z"].null_count():
        raise ValueError(f"{label} windows contain null is_memorized or p_z")

    mem = w.filter(pl.col("is_memorized"))
    return {
        "n_windows_total": w.height,
        "memorized_windows": mem.height,
        "mean_memorized_pz": float(mem["p_z"].mean()) if mem.height else 0.0,
    }


def _side_agg_config(
    cfg: str,
    repo: str,
    forget_ids: set[str],
) -> tuple[str, dict]:
    """One side's per-config payload. Single-sided shape (acc, n) — not (pre, post).

    Returned `kind`:
      - "group": {"group": <bbh|mmlu>, "acc": float, "n": int}
      - "gsm8k": {"rows": {"strict_match": {acc, n}, "flexible_extract": {acc, n}}}
      - "qa":    {"row": {acc, n, "forget": {acc, n}, "retain": {acc, n}}}
      - "row":   {"row": {acc, n}}
    """
    df = _config_to_polars(repo, cfg)
    _check_scores(df, repo, cfg)

    group = next((g for g, pred in _GROUPS.items() if pred(cfg)), None)
    if group is not None:
        return cfg, {
            "kind": "group",
            "group": group,
            "acc": float(df["score"].mean()),
            "n": df.height,
        }

    if cfg == "gsm8k":
        _require_columns(df, ["filter"], f"{repo}/gsm8k")
        rows: dict[str, dict] = {}
        for filt in ("strict-match", "flexible-extract"):
            f = df.filter(pl.col("filter") == filt)
            if f.is_empty():
                raise ValueError(f"{repo}/gsm8k missing rows for filter={filt!r}")
            rows[filt.replace("-", "_")] = {"acc": float(f["score"].mean()), "n": f.height}
        return cfg, {"kind": "gsm8k", "rows": rows}

    if cfg == "cleanslate_qa":
        _require_columns(df, ["doc"], f"{repo}/cleanslate_qa")
        q = df.with_columns(pl.col("doc").struct.field("content_id").alias("_cid"))
        if q["_cid"].null_count():
            raise ValueError(f"{repo}/cleanslate_qa rows contain null doc.content_id")
        forget_list = list(forget_ids)
        forget_q = q.filter(pl.col("_cid").is_in(forget_list))
        retain_q = q.filter(~pl.col("_cid").is_in(forget_list))
        return cfg, {
            "kind": "qa",
            "row": {
                "acc": float(q["score"].mean()),
                "n": q.height,
                "forget": {
                    "acc": float(forget_q["score"].mean()) if forget_q.height else 0.0,
                    "n": forget_q.height,
                },
                "retain": {
                    "acc": float(retain_q["score"].mean()) if retain_q.height else 0.0,
                    "n": retain_q.height,
                },
            },
        }

    return cfg, {
        "kind": "row",
        "row": {"acc": float(df["score"].mean()), "n": df.height},
    }


# ---------------------------------------------------------------------------
# Retention groups + small helpers
# ---------------------------------------------------------------------------


_GROUPS = {
    "bbh": lambda cfg: cfg.startswith("bbh_cot_fewshot_"),
    "mmlu": lambda cfg: cfg.startswith("mmlu_"),
}


def _two_sided_row(pre: float, post: float, n: int) -> dict:
    return {"pre": pre, "post": post, "delta": post - pre, "n": n}


# ---------------------------------------------------------------------------
# Two-sided assemblers (built on the per-side primitives)
# ---------------------------------------------------------------------------


def _mem_block(
    universe: set[str],
    pre: pl.DataFrame,
    post: pl.DataFrame,
) -> dict:
    """Two-sided window aggregates: pre vs post over `universe`."""
    p = _side_mem(universe, pre, "pre-eval")
    q = _side_mem(universe, post, "post-eval")
    return {
        "n_content": len(universe),
        "n_windows_total": p["n_windows_total"],
        "pre_memorized_windows": p["memorized_windows"],
        "post_memorized_windows": q["memorized_windows"],
        "delta_memorized_windows": q["memorized_windows"] - p["memorized_windows"],
        "pre_mean_memorized_pz": p["mean_memorized_pz"],
        "post_mean_memorized_pz": q["mean_memorized_pz"],
    }


def _agg_config(
    cfg: str,
    pre_repo: str,
    post_repo: str,
    forget_ids: set[str],
) -> tuple[str, dict]:
    """Two-sided per-config payload built from two `_side_agg_config` calls."""
    _, pp = _side_agg_config(cfg, pre_repo, forget_ids)
    _, qq = _side_agg_config(cfg, post_repo, forget_ids)
    if pp["kind"] != qq["kind"]:
        raise AssertionError(f"{cfg}: payload kind mismatch {pp['kind']!r} vs {qq['kind']!r}")
    kind = pp["kind"]

    if kind == "group":
        return cfg, {
            "kind": "group",
            "group": pp["group"],
            "pre": pp["acc"],
            "post": qq["acc"],
            "n": pp["n"],
        }

    if kind == "gsm8k":
        merged: dict = {"kind": "gsm8k", "rows": {}}
        for filt in pp["rows"]:
            pr = pp["rows"][filt]
            qr = qq["rows"][filt]
            merged["rows"][filt] = _two_sided_row(pr["acc"], qr["acc"], pr["n"])
        return cfg, merged

    if kind == "qa":
        pr = pp["row"]
        qr = qq["row"]
        row = _two_sided_row(pr["acc"], qr["acc"], pr["n"])
        row["forget"] = _two_sided_row(pr["forget"]["acc"], qr["forget"]["acc"], pr["forget"]["n"])
        row["retain"] = _two_sided_row(pr["retain"]["acc"], qr["retain"]["acc"], pr["retain"]["n"])
        return cfg, {"kind": "qa", "row": row}

    pr = pp["row"]
    qr = qq["row"]
    return cfg, {"kind": "row", "row": _two_sided_row(pr["acc"], qr["acc"], pr["n"])}


def _retention_block(
    pre_repo: str,
    post_repo: str,
    *,
    forget_ids: set[str],
    max_workers: int = 16,
) -> dict:
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
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            ex.map(lambda c: _agg_config(c, pre_repo, post_repo, forget_ids), configs)
        )

    out: dict = {}
    grouped: dict[str, list[tuple[float, float, int]]] = {g: [] for g in _GROUPS}
    for cfg, payload in results:
        kind = payload["kind"]
        if kind == "group":
            grouped[payload["group"]].append((payload["pre"], payload["post"], payload["n"]))
        elif kind == "gsm8k":
            for filt, row in payload["rows"].items():
                out[f"gsm8k_{filt}"] = row
        elif kind in ("row", "qa"):
            out[cfg] = payload["row"]
        else:
            raise AssertionError(f"unknown payload kind: {kind!r}")

    for label, subs in grouped.items():
        if not subs:
            continue
        eq_pre = sum(s[0] for s in subs) / len(subs)
        eq_post = sum(s[1] for s in subs) / len(subs)
        n_samples = sum(s[2] for s in subs)
        row = _two_sided_row(eq_pre, eq_post, n_samples)
        row["n_subtasks"] = len(subs)
        row["n_samples"] = n_samples
        out[label] = row

    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# One-sided assembler (consumes _side_agg_config payloads directly)
# ---------------------------------------------------------------------------


def _assemble_one_sided_retention(results: list[tuple[str, dict]]) -> dict:
    out: dict = {}
    grouped: dict[str, list[tuple[float, int]]] = {g: [] for g in _GROUPS}
    for cfg, payload in results:
        kind = payload["kind"]
        if kind == "group":
            grouped[payload["group"]].append((payload["acc"], payload["n"]))
        elif kind == "gsm8k":
            for filt, row in payload["rows"].items():
                out[f"gsm8k_{filt}"] = row
        elif kind in ("row", "qa"):
            out[cfg] = payload["row"]
        else:
            raise AssertionError(f"unknown payload kind: {kind!r}")

    for label, subs in grouped.items():
        if not subs:
            continue
        out[label] = {
            "acc": sum(s[0] for s in subs) / len(subs),
            "n": sum(s[1] for s in subs),
            "n_subtasks": len(subs),
        }

    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_score(
    *,
    forget_pool: str | Path,
    pre_eval: str | Path,
    post_eval: str | Path,
    pre_generations: str,
    post_generations: str,
    checkpoint: str | None = None,
) -> dict:
    """Two-sided pre→post diff score."""
    forget_df = _load_polars(forget_pool)
    pre_df = _load_polars(pre_eval)
    post_df = _load_polars(post_eval)

    forget_ids = _content_id_set(forget_df, label="forget_pool")
    pre_ids = _content_id_set(pre_df, label="pre-eval")
    post_ids = _content_id_set(post_df, label="post-eval")

    shared = pre_ids & post_ids
    if not shared:
        raise ValueError("pre-eval and post-eval share no content_ids")
    missing_forget = forget_ids - shared
    if missing_forget:
        raise ValueError(
            f"{len(missing_forget)} forget content_id(s) missing from the eval universe: "
            f"{sorted(missing_forget)[:5]}"
        )

    forget_in_universe = forget_ids & shared
    retain_ids = shared - forget_ids

    return {
        "forget": _mem_block(forget_in_universe, pre_df, post_df),
        "retain": _mem_block(retain_ids, pre_df, post_df),
        "retention": _retention_block(
            str(pre_generations), str(post_generations), forget_ids=forget_ids
        ),
        "meta": {
            "forget_set": str(forget_pool),
            "pre_eval": str(pre_eval),
            "post_eval": str(post_eval),
            "pre_gen": str(pre_generations),
            "post_gen": str(post_generations),
            "checkpoint": checkpoint,
        },
    }


def _compute_one_sided(
    *,
    forget_pool: str | Path,
    eval_path: str | Path,
    generations: str,
    label: str,
    checkpoint: str | None = None,
    max_workers: int = 16,
) -> dict:
    """Single-side score (pre OR post). `label` is "pre" or "post"."""
    forget_df = _load_polars(forget_pool)
    eval_df = _load_polars(eval_path)

    forget_ids = _content_id_set(forget_df, label="forget_pool")
    eval_ids = _content_id_set(eval_df, label=f"{label}-eval")
    missing = forget_ids - eval_ids
    if missing:
        raise ValueError(
            f"{len(missing)} forget content_id(s) missing from {label}-eval: "
            f"{sorted(missing)[:5]}"
        )
    retain_ids = eval_ids - forget_ids

    forget_block = {
        "n_content": len(forget_ids),
        **_side_mem(forget_ids, eval_df, f"{label}-eval"),
    }
    retain_block = {
        "n_content": len(retain_ids),
        **_side_mem(retain_ids, eval_df, f"{label}-eval"),
    }

    configs = sorted(get_dataset_config_names(generations))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(
            ex.map(lambda c: _side_agg_config(c, generations, forget_ids), configs)
        )
    retention = _assemble_one_sided_retention(results)

    meta: dict = {
        "forget_set": str(forget_pool),
        f"{label}_eval": str(eval_path),
        f"{label}_gen": str(generations),
    }
    if checkpoint is not None:
        meta["checkpoint"] = checkpoint

    return {
        "forget": forget_block,
        "retain": retain_block,
        "retention": retention,
        "meta": meta,
    }


def compute_score_pre(
    *,
    forget_pool: str | Path,
    pre_eval: str | Path,
    pre_generations: str,
) -> dict:
    """Pre-side only: baseline summary before any unlearning."""
    return _compute_one_sided(
        forget_pool=forget_pool,
        eval_path=pre_eval,
        generations=pre_generations,
        label="pre",
    )


def compute_score_post(
    *,
    forget_pool: str | Path,
    post_eval: str | Path,
    post_generations: str,
    checkpoint: str | None = None,
) -> dict:
    """Post-side only: spot-check a freshly-trained checkpoint."""
    return _compute_one_sided(
        forget_pool=forget_pool,
        eval_path=post_eval,
        generations=post_generations,
        label="post",
        checkpoint=checkpoint,
    )


def save_score(score: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2))
    return out
