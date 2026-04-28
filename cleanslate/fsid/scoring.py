"""Post-unlearning scoring: compare pre/post memorization + retention artifacts.

Output is a nested dict (saved as JSON):

  forget    -- window-level memorization aggregates over forget content_ids
  retain    -- same aggregates over retain content_ids (eval universe minus forget)
  retention -- per-task lm-eval accuracies (pre, post, delta, retention)
               with BBH/MMLU folded equal-weighted, GSM8K split by filter,
               cleanslate_qa carrying a forget/retain sub-split inside its row
  meta      -- echo of input paths
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from datasets import get_dataset_config_names, load_dataset


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


def _mem_block(
    universe: set[str],
    pre: pl.DataFrame,
    post: pl.DataFrame,
) -> dict:
    """Window-level pre/post aggregates over `universe` (content_ids).

    Explodes `windows` once on each side, filters to `universe`, and aggregates
    via polars. Fails if any content_id in the universe is missing from either
    side, or if any window has null `is_memorized` / `p_z`.
    """
    _require_columns(pre, ["content_id", "windows"], "pre-eval")
    _require_columns(post, ["content_id", "windows"], "post-eval")

    if not universe:
        raise ValueError("empty universe")

    universe_list = list(universe)
    pre_f = pre.filter(pl.col("content_id").is_in(universe_list)).select(["content_id", "windows"])
    post_f = post.filter(pl.col("content_id").is_in(universe_list)).select(["content_id", "windows"])

    pre_ids = set(pre_f["content_id"].to_list())
    post_ids = set(post_f["content_id"].to_list())
    missing_pre = universe - pre_ids
    missing_post = universe - post_ids
    if missing_pre or missing_post:
        raise ValueError(
            f"{len(missing_pre)} forget/retain content_id(s) missing from pre-eval, "
            f"{len(missing_post)} from post-eval. "
            f"pre-missing[:5]={sorted(missing_pre)[:5]}; post-missing[:5]={sorted(missing_post)[:5]}"
        )

    pre_w = pre_f.explode("windows").unnest("windows")
    post_w = post_f.explode("windows").unnest("windows")
    _require_columns(pre_w, ["is_memorized", "p_z"], "pre-eval windows")
    _require_columns(post_w, ["is_memorized", "p_z"], "post-eval windows")
    if pre_w["is_memorized"].null_count() or pre_w["p_z"].null_count():
        raise ValueError("pre-eval windows contain null is_memorized or p_z")
    if post_w["is_memorized"].null_count() or post_w["p_z"].null_count():
        raise ValueError("post-eval windows contain null is_memorized or p_z")

    n_windows_total = pre_w.height
    pre_mem_df = pre_w.filter(pl.col("is_memorized"))
    post_mem_df = post_w.filter(pl.col("is_memorized"))
    pre_mem = pre_mem_df.height
    post_mem = post_mem_df.height
    pre_mean = float(pre_mem_df["p_z"].mean()) if pre_mem else 0.0
    post_mean = float(post_mem_df["p_z"].mean()) if post_mem else 0.0

    return {
        "n_content": len(universe),
        "n_windows_total": n_windows_total,
        "pre_memorized_windows": pre_mem,
        "post_memorized_windows": post_mem,
        "delta_memorized_windows": post_mem - pre_mem,
        "pre_mean_memorized_pz": pre_mean,
        "post_mean_memorized_pz": post_mean,
    }


# ---------------------------------------------------------------------------
# Retention aggregation
# ---------------------------------------------------------------------------


_GROUPS = {
    "bbh": lambda cfg: cfg.startswith("bbh_cot_fewshot_"),
    "mmlu": lambda cfg: cfg.startswith("mmlu_"),
}


def _row(pre: float, post: float, n: int) -> dict:
    return {
        "pre": pre,
        "post": post,
        "delta": post - pre,
        "n": n,
    }


def _config_to_polars(repo: str, cfg: str) -> pl.DataFrame:
    return load_dataset(repo, cfg, split="train").to_polars()


def _check_scores(df: pl.DataFrame, repo: str, cfg: str) -> None:
    _require_columns(df, ["score"], f"{repo}/{cfg}")
    n_null = df["score"].null_count()
    if n_null:
        raise ValueError(f"{repo}/{cfg} has {n_null} null `score` rows — re-run the judge")


def _agg_config(
    cfg: str,
    pre_repo: str,
    post_repo: str,
    forget_ids: set[str],
) -> tuple[str, dict]:
    """Read pre+post for one config in polars and return ('cfg', payload).

    Payload is one of:
      - {"kind": "row", "row": <_row>}
      - {"kind": "group", "group": <bbh|mmlu>, "pre": <acc>, "post": <acc>, "n": <n>}
      - {"kind": "gsm8k", "rows": {"strict_match": <_row>, "flexible_extract": <_row>}}
      - {"kind": "qa", "row": <_row with .forget/.retain>}
    """
    pre_df = _config_to_polars(pre_repo, cfg)
    post_df = _config_to_polars(post_repo, cfg)
    _check_scores(pre_df, pre_repo, cfg)
    _check_scores(post_df, post_repo, cfg)

    group = next((g for g, pred in _GROUPS.items() if pred(cfg)), None)
    if group is not None:
        return cfg, {
            "kind": "group",
            "group": group,
            "pre": float(pre_df["score"].mean()),
            "post": float(post_df["score"].mean()),
            "n": pre_df.height,
        }

    if cfg == "gsm8k":
        _require_columns(pre_df, ["filter"], f"{pre_repo}/gsm8k")
        _require_columns(post_df, ["filter"], f"{post_repo}/gsm8k")
        out: dict = {"kind": "gsm8k", "rows": {}}
        for filt in ("strict-match", "flexible-extract"):
            pre_f = pre_df.filter(pl.col("filter") == filt)
            post_f = post_df.filter(pl.col("filter") == filt)
            if pre_f.is_empty() or post_f.is_empty():
                raise ValueError(
                    f"gsm8k missing rows for filter={filt!r} "
                    f"(pre={pre_f.height}, post={post_f.height})"
                )
            out["rows"][filt.replace("-", "_")] = _row(
                float(pre_f["score"].mean()),
                float(post_f["score"].mean()),
                pre_f.height,
            )
        return cfg, out

    if cfg == "cleanslate_qa":
        _require_columns(pre_df, ["doc"], f"{pre_repo}/cleanslate_qa")
        _require_columns(post_df, ["doc"], f"{post_repo}/cleanslate_qa")
        # `doc` is a Struct({'content_id': str, ...}); pull the field out.
        pre_q = pre_df.with_columns(
            pl.col("doc").struct.field("content_id").alias("_cid")
        )
        post_q = post_df.with_columns(
            pl.col("doc").struct.field("content_id").alias("_cid")
        )
        if pre_q["_cid"].null_count() or post_q["_cid"].null_count():
            raise ValueError("cleanslate_qa rows contain null doc.content_id")

        forget_list = list(forget_ids)
        pre_forget = pre_q.filter(pl.col("_cid").is_in(forget_list))
        post_forget = post_q.filter(pl.col("_cid").is_in(forget_list))
        pre_retain = pre_q.filter(~pl.col("_cid").is_in(forget_list))
        post_retain = post_q.filter(~pl.col("_cid").is_in(forget_list))

        row = _row(
            float(pre_q["score"].mean()),
            float(post_q["score"].mean()),
            pre_q.height,
        )
        row["forget"] = _row(
            float(pre_forget["score"].mean()) if pre_forget.height else 0.0,
            float(post_forget["score"].mean()) if post_forget.height else 0.0,
            pre_forget.height,
        )
        row["retain"] = _row(
            float(pre_retain["score"].mean()) if pre_retain.height else 0.0,
            float(post_retain["score"].mean()) if post_retain.height else 0.0,
            pre_retain.height,
        )
        return cfg, {"kind": "qa", "row": row}

    return cfg, {
        "kind": "row",
        "row": _row(
            float(pre_df["score"].mean()),
            float(post_df["score"].mean()),
            pre_df.height,
        ),
    }


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
        row = _row(eq_pre, eq_post, n_samples)
        row["n_subtasks"] = len(subs)
        row["n_samples"] = n_samples
        out[label] = row

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
    """Compute the nested scoring dict.

    Inputs accept HF dataset IDs (`org/repo[:config[:split]]`) or local
    parquet files. Every aggregation runs in polars; any missing/null data
    raises rather than degrading silently.
    """
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


def save_score(score: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(score, indent=2))
    return out
