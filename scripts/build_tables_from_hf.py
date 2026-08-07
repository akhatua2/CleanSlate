"""
Build LaTeX rows for the SimNPO-gentle |F|=50 result tables, fetching everything
from HuggingFace (no Excel dependency).

Two-phase design:
  Phase 1 (warm cache): walk every (repo, config) we need, write per-(repo,config)
                        aggregate JSON files to .cache/tables_from_hf/ .
  Phase 2 (emit):       read the cache and emit LaTeX rows for two tables:
                          tab:retrieval — BM25 mid / Infinigram mid / BM25 pre × 6
                          tab:oracle    — Oracle (eval-aware) curator × 6

Run with --warm to populate the cache (idempotent), then with no flags to
re-emit cheaply.

Validation deltas are reported as relative %% change of the baseline score by
default ((post-pre)/pre*100); use --metric pp for absolute pp.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("DATASETS_VERBOSITY", "error")
logging.basicConfig(level=logging.WARNING)

from huggingface_hub import HfApi  # noqa: E402

CACHE_DIR = Path(".cache/tables_from_hf")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Model registry ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Model:
    display: str         # row label in the LaTeX table
    base_clean: str      # used in eval-/generations-{base_clean}
    fsid_short: str      # used in fsid-curated-{fsid_short}
    org_post: str        # which HF org owns the post-unlearning datasets


# org_post for retrieval table: nemotron + gemma live in arpandeepk
MODELS_RETRIEVAL = [
    Model("Llama-3.1-8B",  "llama-3_1-8b",        "llama-8b",     "unlearning-cleanslate"),
    Model("Olmo-3-7B",     "olmo-3-7b",           "olmo-7b",      "unlearning-cleanslate"),
    Model("Qwen3-8B",      "qwen3-8b",            "qwen3-8b",     "unlearning-cleanslate"),
    Model("Nemotron-9B",   "nemotron-nano-9b-v2", "nemotron-9b",  "arpandeepk"),
    Model("Gemma-3-12B",   "gemma-3-12b",         "gemma-12b",    "arpandeepk"),
    Model("Olmo-3-32B",    "olmo-3-32b",          "olmo-32b",     "arpandeepk"),
]

# Oracle baselines all live in unlearning-cleanslate
MODELS_ORACLE = [
    Model("Llama-3.1-8B",  "llama-3_1-8b",        "llama-8b",     "unlearning-cleanslate"),
    Model("Olmo-3-7B",     "olmo-3-7b",           "olmo-7b",      "unlearning-cleanslate"),
    Model("Nemotron-9B",   "nemotron-nano-9b-v2", "nemotron-9b",  "unlearning-cleanslate"),
    Model("Qwen3-8B",      "qwen3-8b",            "qwen3-8b",     "unlearning-cleanslate"),
    Model("Gemma-3-12B",   "gemma-3-12b",         "gemma-12b",    "unlearning-cleanslate"),
    Model("Olmo-3-32B",    "olmo-3-32b",          "olmo-32b",     "unlearning-cleanslate"),
]

RULES_RETRIEVAL = [
    ("BM25 mid",       "bm25_10B", "bm25-10b"),
    ("Infinigram mid", "igm_10B",  "igm-10b"),
    ("BM25 pre",       "bm25_6T",  "bm25-6t"),
]
RULE_ORACLE = ("baseline", "baseline")  # (forget split, repo slug)

# Algorithm comparison table (Oracle, |F|=50, 3 small models × 3 algos).
# Each algo has its own (post_eval / post_gen) suffix: `<algo>-baseline`.
# The forget split inside fsid-curated:forget is shared = "baseline".
ALGOS = [
    ("SimNPO", "simnpo-gentle-baseline"),
    ("UNDIAL", "undial-baseline"),
    ("RMU",    "rmu-baseline"),
]
MODELS_ALGOS = [
    Model("Llama-3.1-8B", "llama-3_1-8b", "llama-8b",  "unlearning-cleanslate"),
    Model("Olmo-3-7B",    "olmo-3-7b",    "olmo-7b",   "unlearning-cleanslate"),
    Model("Qwen3-8B",     "qwen3-8b",     "qwen3-8b",  "unlearning-cleanslate"),
]

# Per-model overrides for paths that don't follow the default naming convention.
# Currently the only outstanding mismatch: llama Oracle generations were
# pushed under a debug name; another file already occupies the standard
# repo name, so we point at the debug repo here.
ORACLE_OVERRIDES: dict[str, dict[str, str]] = {
    "llama-8b": {
        "post_gen": "unlearning-cleanslate/generations-checkpoint-134-debug-checkpoint-134-llama",
    },
}

VAL_COLS = [
    ("GSM8K",  "gsm8k_flexible_extract"),
    ("BBH",    "bbh"),
    ("HSwag",  "hellaswag"),
    ("WinoG",  "winogrande"),
    ("CoQA",   "coqa"),
    ("HEval+", "humaneval_plus"),
    ("LMBDA",  "lambada_openai"),
]

_BBH_PREFIX = "bbh_cot_fewshot_"
_NEEDED_GEN_CONFIGS = {
    "cleanslate_qa", "gsm8k", "hellaswag", "winogrande",
    "coqa", "humaneval_plus", "lambada_openai",
}


# ── HF helpers ────────────────────────────────────────────────────────────────
_API = HfApi()


def _exists(repo: str) -> bool:
    try:
        _API.dataset_info(repo)
        return True
    except Exception:
        return False


def _safe_name(s: str) -> str:
    return s.replace("/", "__").replace(":", "..")


def _cache_path(repo: str, config: str) -> Path:
    return CACHE_DIR / f"{_safe_name(repo)}__{config}.json"


# ── Per-(repo,config) downloaders that write JSON ─────────────────────────────
def _cache_eval(repo: str) -> dict | None:
    """Per-content_id → (num_windows, memorized_windows). Cached as JSON.
    Uses HF Hub parquet API directly + polars to avoid loading the heavy
    per-window column."""
    cache = _cache_path(repo, "eval_per_content")
    if cache.exists():
        return json.loads(cache.read_text())
    if not _exists(repo):
        sys.stderr.write(f"[miss] {repo} not on HF\n")
        return None
    try:
        sys.stderr.write(f"[load] {repo}/default …\n")
        # Use polars + HF auto-resolution: it streams the parquet column-by-column,
        # so we never materialise the windows list in Python.
        import polars as pl
        df = pl.read_parquet(
            f"hf://datasets/{repo}/data/train-*.parquet",
            columns=["content_id", "num_windows", "memorized_windows"],
        )
        per_cid = {
            row["content_id"]: {
                "num_windows": int(row["num_windows"] or 0),
                "memorized_windows": int(row["memorized_windows"] or 0),
            }
            for row in df.iter_rows(named=True)
        }
        cache.write_text(json.dumps(per_cid))
        return per_cid
    except Exception as e:
        sys.stderr.write(f"[err] {repo}: {type(e).__name__}: {str(e)[:200]}\n")
        return None


def _cache_gen_config(repo: str, config: str, *, retries: int = 5) -> dict | None:
    """Per-config aggregate. Uses polars streaming to read only score/filter/doc.
    Retries with exponential backoff on rate-limit / transient errors."""
    import time
    cache = _cache_path(repo, config)
    if cache.exists():
        return json.loads(cache.read_text())
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            import polars as pl
            sys.stderr.write(f"[load] {repo}/{config}\n")
            path = f"hf://datasets/{repo}/{config}/train-*.parquet"
            if config == "cleanslate_qa":
                df = pl.read_parquet(path, columns=["doc", "score"])
                df = df.with_columns(pl.col("doc").struct.field("content_id").alias("cid"))
                rows = [
                    {"cid": r["cid"], "score": float(r["score"])}
                    for r in df.select(["cid", "score"]).iter_rows(named=True)
                ]
                payload = {"kind": "cleanslate_qa", "rows": rows}
            elif config == "gsm8k":
                df = pl.read_parquet(path, columns=["filter", "score"])
                rows = [
                    {"filter": r["filter"], "score": float(r["score"])}
                    for r in df.iter_rows(named=True)
                ]
                payload = {"kind": "gsm8k", "rows": rows}
            else:
                df = pl.read_parquet(path, columns=["score"])
                mean = float(df["score"].mean()) if df.height else 0.0
                payload = {"kind": "mean", "mean": mean, "n": df.height}
            cache.write_text(json.dumps(payload))
            return payload
        except Exception as e:
            last_err = e
            msg = str(e)
            # Retry on rate-limit / transient
            if "rate limit" in msg.lower() or "429" in msg or "503" in msg:
                wait = (2 ** attempt) + 1
                sys.stderr.write(f"[retry {attempt+1}/{retries} in {wait}s] {repo}/{config}\n")
                time.sleep(wait)
                continue
            break
    sys.stderr.write(f"[err] {repo}/{config}: {type(last_err).__name__}: {str(last_err)[:200]}\n")
    return None


def _cache_forget(fsid_repo: str, split: str) -> set[str] | None:
    """Set of forget content_ids from fsid-curated:forget:{split}."""
    cache = _cache_path(fsid_repo, f"forget_{split}")
    if cache.exists():
        return set(json.loads(cache.read_text()))
    try:
        from datasets import load_dataset
        sys.stderr.write(f"[load] {fsid_repo}/forget:{split} …\n")
        ds = load_dataset(fsid_repo, "forget", split=split)
        cids = sorted({r["content_id"] for r in ds if r.get("content_id")})
        cache.write_text(json.dumps(cids))
        return set(cids)
    except Exception as e:
        sys.stderr.write(f"[err] {fsid_repo}:forget:{split}: {type(e).__name__}: {str(e)[:200]}\n")
        return None


# ── Path layout for one cell ──────────────────────────────────────────────────
def _paths_for_algo(model: Model, algo_suffix: str) -> dict[str, str]:
    """Paths for one (model, algorithm) cell of the algorithm comparison table.
    Always uses the 'baseline' forget split. Both eval and generations live in
    unlearning-cleanslate."""
    fsid = f"unlearning-cleanslate/fsid-curated-{model.fsid_short}"
    pre_eval = f"unlearning-cleanslate/eval-{model.base_clean}"
    pre_gen = f"unlearning-cleanslate/generations-{model.base_clean}-pre_val"
    post_eval = f"unlearning-cleanslate/eval-{model.base_clean}-{algo_suffix}"
    post_gen = f"unlearning-cleanslate/generations-{model.base_clean}-{algo_suffix}"
    # SimNPO uses the same overrides as the Oracle table (e.g. llama gen)
    if algo_suffix == "simnpo-gentle-baseline":
        for k, v in ORACLE_OVERRIDES.get(model.fsid_short, {}).items():
            if k == "post_eval":
                post_eval = v
            elif k == "post_gen":
                post_gen = v
    return {
        "fsid": fsid, "rule_split": "baseline",
        "pre_eval": pre_eval, "post_eval": post_eval,
        "pre_gen": pre_gen, "post_gen": post_gen,
    }


def _paths_for(model: Model, rule_split: str, rule_slug: str, *, oracle: bool) -> dict[str, str]:
    fsid = f"unlearning-cleanslate/fsid-curated-{model.fsid_short}"
    pre_eval = f"unlearning-cleanslate/eval-{model.base_clean}"
    pre_gen = f"unlearning-cleanslate/generations-{model.base_clean}-pre_val"
    if oracle:
        post_eval = f"unlearning-cleanslate/eval-{model.base_clean}-simnpo-gentle-baseline"
        post_gen = f"unlearning-cleanslate/generations-{model.base_clean}-simnpo-gentle-baseline"
        for k, v in ORACLE_OVERRIDES.get(model.fsid_short, {}).items():
            if k == "post_eval":
                post_eval = v
            elif k == "post_gen":
                post_gen = v
    else:
        post_eval = f"{model.org_post}/eval-{model.base_clean}-simnpo-gentle-{rule_slug}"
        post_gen = f"{model.org_post}/generations-{model.base_clean}-simnpo-gentle-{rule_slug}"
    return {
        "fsid": fsid, "rule_split": rule_split,
        "pre_eval": pre_eval, "post_eval": post_eval,
        "pre_gen": pre_gen, "post_gen": post_gen,
    }


# ── Phase 1: warm cache ───────────────────────────────────────────────────────
def _gen_configs_to_load(repo: str) -> list[str]:
    """Return the gen-dataset configs we actually use (NEEDED + bbh_*)."""
    if not _exists(repo):
        return []
    from datasets import get_dataset_config_names
    try:
        all_cfgs = set(get_dataset_config_names(repo))
    except Exception as e:
        sys.stderr.write(f"[err] config_names {repo}: {e}\n")
        return []
    keep = [c for c in all_cfgs if c in _NEEDED_GEN_CONFIGS or c.startswith(_BBH_PREFIX)]
    return sorted(keep)


def warm_cache(workers: int = 2) -> None:
    """Walk every cell × every required (repo, config) and ensure cache exists.
    Parallelised with a thread pool — HF downloads are I/O-bound."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cells: list[dict] = []
    for m in MODELS_RETRIEVAL:
        for _, rule_split, rule_slug in RULES_RETRIEVAL:
            cells.append(_paths_for(m, rule_split, rule_slug, oracle=False))
    for m in MODELS_ORACLE:
        rule_split, rule_slug = RULE_ORACLE
        cells.append(_paths_for(m, rule_split, rule_slug, oracle=True))
    # Algo comparison cells
    for m in MODELS_ALGOS:
        for _, suffix in ALGOS:
            cells.append(_paths_for_algo(m, suffix))

    eval_repos: set[str] = set()
    gen_repos: set[str] = set()
    fsid_keys: set[tuple[str, str]] = set()
    for c in cells:
        eval_repos.add(c["pre_eval"]); eval_repos.add(c["post_eval"])
        gen_repos.add(c["pre_gen"]);   gen_repos.add(c["post_gen"])
        fsid_keys.add((c["fsid"], c["rule_split"]))

    print(f"warm cache: {len(eval_repos)} eval repos, {len(gen_repos)} gen repos, {len(fsid_keys)} forget pools")

    # 1. forget pools (parallel)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda kv: _cache_forget(*kv), sorted(fsid_keys)))

    # 2. eval per-content (parallel)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_cache_eval, sorted(eval_repos)))

    # 3. generations (parallel — most expensive phase)
    gen_jobs: list[tuple[str, str]] = []
    for repo in sorted(gen_repos):
        for cfg in _gen_configs_to_load(repo):
            gen_jobs.append((repo, cfg))
    print(f"  gen: {len(gen_jobs)} (repo, config) pairs to load")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_cache_gen_config, r, c): (r, c) for r, c in gen_jobs}
        for f in as_completed(futs):
            done += 1
            if done % 25 == 0:
                print(f"  gen: {done}/{len(gen_jobs)}")

    print("warm cache: done")


# ── Phase 2: read cache and assemble row cells ────────────────────────────────
def _mem_block(per_cid: dict | None, ids: set[str]) -> tuple[int, int] | None:
    if per_cid is None:
        return None
    n_w = mem_w = 0
    for cid in ids:
        rec = per_cid.get(cid)
        if rec:
            n_w += rec["num_windows"]; mem_w += rec["memorized_windows"]
    return n_w, mem_w


def _aggregate_qa(payload: dict | None, retain_ids: set[str]) -> tuple[float, int] | None:
    if not payload or payload["kind"] != "cleanslate_qa":
        return None
    scores = [r["score"] for r in payload["rows"] if r["cid"] in retain_ids]
    if not scores:
        return None
    return sum(scores) / len(scores), len(scores)


def _aggregate_gsm8k(payload: dict | None) -> tuple[float, int] | None:
    if not payload or payload["kind"] != "gsm8k":
        return None
    scores = [r["score"] for r in payload["rows"] if r.get("filter") == "flexible-extract"]
    if not scores:
        return None
    return sum(scores) / len(scores), len(scores)


def _aggregate_mean(payload: dict | None) -> tuple[float, int] | None:
    if not payload or payload["kind"] != "mean":
        return None
    return payload["mean"], payload["n"]


def _retention_from_cache(pre_repo: str, post_repo: str, retain_ids: set[str]) -> dict:
    """Build retention dict for the columns we display."""
    out: dict[str, dict] = {}

    # cleanslate_qa (restricted to retain ids)
    pre = _aggregate_qa(_load_cache(pre_repo, "cleanslate_qa"), retain_ids)
    post = _aggregate_qa(_load_cache(post_repo, "cleanslate_qa"), retain_ids)
    if pre and post:
        out["cleanslate_qa"] = {"pre": pre[0], "post": post[0], "delta": post[0] - pre[0], "n": pre[1]}

    # gsm8k (flexible extract)
    pre = _aggregate_gsm8k(_load_cache(pre_repo, "gsm8k"))
    post = _aggregate_gsm8k(_load_cache(post_repo, "gsm8k"))
    if pre and post:
        out["gsm8k_flexible_extract"] = {"pre": pre[0], "post": post[0], "delta": post[0] - pre[0], "n": pre[1]}

    # single-config tasks
    for cfg in ("hellaswag", "winogrande", "coqa", "humaneval_plus", "lambada_openai"):
        pre = _aggregate_mean(_load_cache(pre_repo, cfg))
        post = _aggregate_mean(_load_cache(post_repo, cfg))
        if pre and post:
            out[cfg] = {"pre": pre[0], "post": post[0], "delta": post[0] - pre[0], "n": pre[1]}

    # bbh — average across all bbh_* sub-configs present in BOTH
    pre_files = sorted(CACHE_DIR.glob(f"{_safe_name(pre_repo)}__{_BBH_PREFIX}*.json"))
    post_avail = {p.name for p in CACHE_DIR.glob(f"{_safe_name(post_repo)}__{_BBH_PREFIX}*.json")}
    pre_means: list[float] = []; post_means: list[float] = []
    for p in pre_files:
        sub = p.name.split("__")[-1].removesuffix(".json")
        post_p = _safe_name(post_repo) + "__" + sub + ".json"
        if post_p in post_avail:
            pre_p = _aggregate_mean(_load_cache(pre_repo, sub))
            post_p_v = _aggregate_mean(_load_cache(post_repo, sub))
            if pre_p and post_p_v:
                pre_means.append(pre_p[0]); post_means.append(post_p_v[0])
    if pre_means:
        pre_b = sum(pre_means) / len(pre_means)
        post_b = sum(post_means) / len(post_means)
        out["bbh"] = {"pre": pre_b, "post": post_b, "delta": post_b - pre_b, "n_subtasks": len(pre_means)}

    return out


def _load_cache(repo: str, key: str) -> dict | None:
    p = _cache_path(repo, key)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _build_score(model: Model, rule_split: str, rule_slug: str, *, oracle: bool) -> dict | None:
    """Best-effort: returns whatever can be computed.
    - If forget_pool + pre_eval + post_eval present → forget/retain memorization.
    - If pre_gen + post_gen present → retention block.
    Either alone (or both) is enough to emit the row; missing parts → "--"."""
    paths = _paths_for(model, rule_split, rule_slug, oracle=oracle)
    forget_ids = _cache_forget(paths["fsid"], paths["rule_split"])
    if forget_ids is None:
        return None  # without forget split we can't split retain QA either

    score: dict = {}
    pre_per = _cache_eval(paths["pre_eval"])
    post_per = _cache_eval(paths["post_eval"])
    if pre_per is not None and post_per is not None:
        all_ids = set(pre_per.keys())
        retain_ids = all_ids - forget_ids
        pre_f = _mem_block(pre_per, forget_ids); post_f = _mem_block(post_per, forget_ids)
        pre_r = _mem_block(pre_per, retain_ids); post_r = _mem_block(post_per, retain_ids)
        if all([pre_f, post_f, pre_r, post_r]):
            score["forget"] = {"pre": pre_f[1], "post": post_f[1]}
            score["retain"] = {"pre": pre_r[1], "post": post_r[1]}

    # Retention only needs the gen datasets. Use the union of forget/retain sets
    # we can actually derive: if pre-eval is present, retain = all-forget; else
    # restrict the cleanslate_qa "retain" column to "anything not in forget_ids".
    retain_ids_for_qa = (
        set(pre_per.keys()) - forget_ids if pre_per is not None else set()
    )
    if pre_per is None:
        # Best-effort: pull content_ids from pre_gen's cleanslate_qa cache
        qa = _load_cache(paths["pre_gen"], "cleanslate_qa")
        if qa and qa.get("kind") == "cleanslate_qa":
            retain_ids_for_qa = {r["cid"] for r in qa["rows"] if r["cid"]} - forget_ids
    retention = _retention_from_cache(paths["pre_gen"], paths["post_gen"], retain_ids_for_qa)
    if retention:
        score["retention"] = retention
    if not score:
        return None
    score.setdefault("forget", {"pre": None, "post": None})
    score.setdefault("retain", {"pre": None, "post": None})
    return score


# ── LaTeX formatting ──────────────────────────────────────────────────────────
# Color stops for the rich (--rich) format. Cell colors are picked based on
# magnitude / sign: greens for "improvement", reds for "degradation".
_GREEN = ["cg25", "cg50", "cg75", "cg100"]
_RED   = ["cr25", "cr50", "cr75", "cr100"]


def _color_delta_pp(delta_pp: float) -> str:
    """Color a (pp) delta on a 5-stop scale: bigger magnitude = saturated color."""
    if delta_pp == 0 or abs(delta_pp) < 0.05:
        return "cneutral"
    bucket = min(3, int(abs(delta_pp) // 1.5))  # |Δ|<1.5→0; <3→1; <4.5→2; ≥4.5→3
    return _GREEN[bucket] if delta_pp > 0 else _RED[bucket]


def _color_delta_pctchg(delta_pct: float) -> str:
    """Color a relative %% change on a 5-stop scale (larger magnitudes than pp)."""
    if delta_pct == 0 or abs(delta_pct) < 0.5:
        return "cneutral"
    if abs(delta_pct) < 5:
        bucket = 0
    elif abs(delta_pct) < 15:
        bucket = 1
    elif abs(delta_pct) < 30:
        bucket = 2
    else:
        bucket = 3
    return _GREEN[bucket] if delta_pct > 0 else _RED[bucket]


def _color_F(pct: float) -> str:
    """F (forget unmem %): higher = greener. pct is in [-100, 100]."""
    if abs(pct) < 1:
        return "cneutral"
    bucket = min(3, int(abs(pct) // 25))
    return _GREEN[bucket] if pct > 0 else _RED[bucket]


def _color_R(pct: float) -> str:
    """R (retain unmem %): lower = greener (less collateral). pct in [0, 100]+."""
    if pct < 1:
        return "cg100"
    bucket = min(3, int(pct // 25))
    return _RED[bucket]


def _format_rich_delta(row: dict, metric: str = "pctchg") -> str:
    """Just the Δ number (pctchg or pp) with a cellcolor based on its magnitude."""
    pre, post = row.get("pre"), row.get("post")
    if pre is None or post is None:
        return "--"
    if metric == "pp":
        delta = (post - pre) * 100
        color = _color_delta_pp(delta)
    elif metric == "pctchg":
        if pre == 0:
            return "--"
        delta = (post - pre) / pre * 100
        color = _color_delta_pctchg(delta)
    else:
        raise ValueError(metric)
    return f"\\cellcolor{{{color}}} {delta:+.1f}"


def _format_rich_F(pct: float) -> str:
    return f"\\cellcolor{{{_color_F(pct)}}} {pct:.1f}\\%"


def _format_rich_R(pct: float) -> str:
    return f"\\cellcolor{{{_color_R(pct)}}} {pct:.1f}\\%"


def _format_delta(row: dict, metric: str) -> str:
    pre, post = row.get("pre"), row.get("post")
    if pre is None or post is None:
        return "--"
    if metric == "pp":
        return f"{(post - pre) * 100:+.1f}"
    if metric == "pctchg":
        if pre == 0:
            return "--"
        return f"{(post - pre) / pre * 100:+.1f}"
    raise ValueError(metric)


def _avg_cells(scores: list[dict], metric: str, *, rich: bool = False) -> dict[str, str]:
    """Average each metric across the section's scores; render with `cavg` color
    in rich mode (or just the mean with no decoration in plain mode)."""
    keys = ("F", "R", "QA", *[c for c, _ in VAL_COLS], "AvgVal")
    out: dict[str, str] = {k: "--" for k in keys}
    if not scores:
        return out

    def _avg(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    # F: mean of F% across scores that have forget mem block
    f_vals: list[float] = []
    r_vals: list[float] = []
    for s in scores:
        f = s.get("forget", {})
        if f.get("pre"):
            f_vals.append((f["pre"] - f["post"]) / f["pre"] * 100)
        r = s.get("retain", {})
        if r.get("pre"):
            r_vals.append((r["pre"] - r["post"]) / r["pre"] * 100)
    f_avg = _avg(f_vals); r_avg = _avg(r_vals)
    if f_avg is not None:
        out["F"] = (f"\\cellcolor{{cavg}} {f_avg:.1f}\\%" if rich else f"{f_avg:.1f}")
    if r_avg is not None:
        out["R"] = (f"\\cellcolor{{cavg}} {r_avg:.1f}\\%" if rich else f"{r_avg:.1f}")

    # QA + validation deltas: average using `metric`
    def _delta_for(row: dict) -> float | None:
        pre, post = row.get("pre"), row.get("post")
        if pre is None or post is None:
            return None
        if metric == "pp":
            return (post - pre) * 100
        if pre == 0:
            return None
        return (post - pre) / pre * 100

    for label, key in (("QA", "cleanslate_qa"), *VAL_COLS):
        deltas = []
        for s in scores:
            row = s.get("retention", {}).get(key)
            d = _delta_for(row) if row else None
            if d is not None:
                deltas.append(d)
        avg = _avg(deltas)
        if avg is not None:
            out[label] = (f"\\cellcolor{{cavg}} {avg:+.1f}" if rich else f"{avg:+.1f}")

    # AvgVal: average of (mean validation delta) per model. Each model
    # contributes its own per-row validation mean (matches the AvgVal column
    # produced by `_row_cells`).
    per_model_means: list[float] = []
    for s in scores:
        ds = []
        for _, key in VAL_COLS:
            row = s.get("retention", {}).get(key)
            d = _delta_for(row) if row else None
            if d is not None:
                ds.append(d)
        if ds:
            per_model_means.append(sum(ds) / len(ds))
    if per_model_means:
        avg = sum(per_model_means) / len(per_model_means)
        out["AvgVal"] = (f"\\cellcolor{{cavg}} {avg:+.1f}" if rich else f"{avg:+.1f}")
    return out


def _row_cells(score: dict | None, metric: str, *, rich: bool = False) -> dict[str, str]:
    out = {k: "--" for k in ("F", "R", "QA", *[c for c, _ in VAL_COLS], "AvgVal")}
    if not score:
        return out
    f = score.get("forget", {})
    if f.get("pre"):
        f_pct = (f["pre"] - f["post"]) / f["pre"] * 100
        out["F"] = _format_rich_F(f_pct) if rich else f"{f_pct:.1f}"
    r = score.get("retain", {})
    if r.get("pre"):
        r_pct = (r["pre"] - r["post"]) / r["pre"] * 100
        out["R"] = _format_rich_R(r_pct) if rich else f"{r_pct:.1f}"
    ret = score.get("retention", {})
    qa = ret.get("cleanslate_qa")
    if qa:
        out["QA"] = _format_rich_delta(qa, metric) if rich else _format_delta(qa, metric)

    val_deltas: list[float] = []
    for label, key in VAL_COLS:
        row = ret.get(key)
        if row:
            out[label] = _format_rich_delta(row, metric) if rich else _format_delta(row, metric)
            pre, post = row.get("pre"), row.get("post")
            if pre is not None and post is not None:
                if metric == "pp":
                    val_deltas.append((post - pre) * 100)
                elif pre != 0:
                    val_deltas.append((post - pre) / pre * 100)
    if val_deltas:
        avg = sum(val_deltas) / len(val_deltas)
        if rich:
            color = _color_delta_pp(avg) if metric == "pp" else _color_delta_pctchg(avg)
            out["AvgVal"] = f"\\cellcolor{{{color}}} {avg:+.1f}"
        else:
            out["AvgVal"] = f"{avg:+.1f}"
    return out


def emit_retrieval(metric: str, *, rich: bool = False) -> None:
    blocks = []
    for label, rule_split, rule_slug in RULES_RETRIEVAL:
        # Use multirow=7 (6 models + 1 average) when averages are added
        lines = [f"\\multirow{{7}}{{*}}{{{label}}}"]
        section_scores: list[dict] = []
        for m in MODELS_RETRIEVAL:
            score = _build_score(m, rule_split, rule_slug, oracle=False)
            if score:
                section_scores.append(score)
            cells = _row_cells(score, metric, rich=rich)
            row = f"& {m.display:<14}"
            row += f" & {cells['F']:>5} & {cells['R']:>5} & {cells['QA']:>5}"
            for label_v, _ in VAL_COLS:
                row += f" & {cells[label_v]:>5}"
            row += f" & {cells['AvgVal']:>5}"
            row += " \\\\"
            lines.append(row)
        # Average row
        avg_cells = _avg_cells(section_scores, metric, rich=rich)
        avg = f"& \\textit{{Average}}    "
        avg += f" & {avg_cells['F']:>5} & {avg_cells['R']:>5} & {avg_cells['QA']:>5}"
        for label_v, _ in VAL_COLS:
            avg += f" & {avg_cells[label_v]:>5}"
        avg += f" & {avg_cells['AvgVal']:>5}"
        avg += " \\\\"
        lines.append(avg)
        blocks.append("\n".join(lines))
    print("% --- Retrieval table (BM25 mid / Infinigram mid / BM25 pre) ---")
    print("\n\\midrule\n".join(blocks))


def emit_algos(metric: str, *, rich: bool = False) -> None:
    """Algorithm comparison table: SimNPO / UNDIAL / RMU × 3 small models."""
    blocks = []
    for algo_label, suffix in ALGOS:
        # 3 models + 1 average → multirow=4
        lines = [f"\\multirow{{4}}{{*}}{{{algo_label}}}"]
        section_scores: list[dict] = []
        for m in MODELS_ALGOS:
            paths = _paths_for_algo(m, suffix)
            forget_ids = _cache_forget(paths["fsid"], "baseline")
            score = None
            if forget_ids is not None:
                pre_per = _cache_eval(paths["pre_eval"])
                post_per = _cache_eval(paths["post_eval"])
                if pre_per is not None and post_per is not None:
                    all_ids = set(pre_per.keys())
                    retain_ids = all_ids - forget_ids
                    pre_f = _mem_block(pre_per, forget_ids); post_f = _mem_block(post_per, forget_ids)
                    pre_r = _mem_block(pre_per, retain_ids); post_r = _mem_block(post_per, retain_ids)
                    if all([pre_f, post_f, pre_r, post_r]):
                        retention = _retention_from_cache(paths["pre_gen"], paths["post_gen"], retain_ids)
                        score = {
                            "forget": {"pre": pre_f[1], "post": post_f[1]},
                            "retain": {"pre": pre_r[1], "post": post_r[1]},
                            "retention": retention,
                        }
            if score:
                section_scores.append(score)
            cells = _row_cells(score, metric, rich=rich)
            row = f"& {m.display:<14}"
            row += f" & {cells['F']:>5} & {cells['R']:>5} & {cells['QA']:>5}"
            for label_v, _ in VAL_COLS:
                row += f" & {cells[label_v]:>5}"
            row += f" & {cells['AvgVal']:>5}"
            row += " \\\\"
            lines.append(row)
        # Average row
        avg_cells = _avg_cells(section_scores, metric, rich=rich)
        avg = f"& \\textit{{Average}}    "
        avg += f" & {avg_cells['F']:>5} & {avg_cells['R']:>5} & {avg_cells['QA']:>5}"
        for label_v, _ in VAL_COLS:
            avg += f" & {avg_cells[label_v]:>5}"
        avg += f" & {avg_cells['AvgVal']:>5}"
        avg += " \\\\"
        lines.append(avg)
        blocks.append("\n".join(lines))
    print("% --- Algorithm comparison table (SimNPO / UNDIAL / RMU) ---")
    print("\n\\midrule\n".join(blocks))


def emit_oracle(metric: str, *, rich: bool = False) -> None:
    print("% --- Oracle table ---")
    rule_split, rule_slug = RULE_ORACLE
    section_scores: list[dict] = []
    for m in MODELS_ORACLE:
        score = _build_score(m, rule_split, rule_slug, oracle=True)
        if score:
            section_scores.append(score)
        cells = _row_cells(score, metric, rich=rich)
        row = f"{m.display:<14}"
        row += f" & {cells['F']:>5} & {cells['R']:>5} & {cells['QA']:>5}"
        for label_v, _ in VAL_COLS:
            row += f" & {cells[label_v]:>5}"
        row += f" & {cells['AvgVal']:>5}"
        row += " \\\\"
        print(row)
    avg_cells = _avg_cells(section_scores, metric, rich=rich)
    avg = f"\\textit{{Average}}     "
    avg += f" & {avg_cells['F']:>5} & {avg_cells['R']:>5} & {avg_cells['QA']:>5}"
    for label_v, _ in VAL_COLS:
        avg += f" & {avg_cells[label_v]:>5}"
    avg += f" & {avg_cells['AvgVal']:>5}"
    avg += " \\\\"
    print(avg)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--warm", action="store_true",
                    help="Phase 1: download every (repo,config) we need into the cache.")
    ap.add_argument("--table", choices=("retrieval", "oracle", "algos", "both", "all"), default="both")
    ap.add_argument("--metric", choices=("pp", "pctchg"), default="pctchg",
                    help="pp = absolute Δ in pp; pctchg = relative %% change. Default: pctchg.")
    ap.add_argument("--rich", action="store_true",
                    help="Emit rich cells with \\cellcolor + pre→post(Δpp) format.")
    ap.add_argument("--clear-cache", action="store_true", help="Wipe cache before warming.")
    args = ap.parse_args()

    if args.clear_cache:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()

    if args.warm:
        warm_cache()
        return 0

    if args.table in ("retrieval", "both", "all"):
        emit_retrieval(args.metric, rich=args.rich)
        if args.table != "retrieval":
            print()
    if args.table in ("oracle", "both", "all"):
        emit_oracle(args.metric, rich=args.rich)
        if args.table == "all":
            print()
    if args.table in ("algos", "all"):
        emit_algos(args.metric, rich=args.rich)
    return 0


if __name__ == "__main__":
    sys.exit(main())
