#!/data/animesh_jha/CleanSlate/.venv/bin/python
"""Prepare exact latest-checkpoint local dirs for final post eval/val.

Login-node safe: uses Hugging Face metadata and snapshot downloads only. No GPU
work is performed here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache" / "post_eval_checkpoints"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "final_final"
DEFAULT_HF_ORG = "unlearning-cleanslate"


@dataclass(frozen=True)
class RowSpec:
    row_id: int
    model: str
    curation: str
    data_source: str
    unlearning: str
    setting: str
    checkpoint_repo: str
    hf_datapath: str


ROWS: tuple[RowSpec, ...] = (
    RowSpec(1, "llama", "Oracle", "-", "Simnpo", "gentle", "unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-baseline-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(2, "qwen3 8b", "Oracle", "-", "Simnpo", "gentle", "unlearning-cleanslate/qwen3-8b-simnpo-gentle-baseline-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
    RowSpec(3, "gemma", "Oracle", "-", "Simnpo", "gentle", "unlearning-cleanslate/gemma-3-12b-simnpo-gentle-baseline-target-100", "unlearning-cleanslate/fsid-curated-gemma-12b-target-100"),
    RowSpec(4, "gemma", "Oracle", "-", "Simnpo", "aggressive", "unlearning-cleanslate/gemma-3-12b-simnpo-baseline-target-100", "unlearning-cleanslate/fsid-curated-gemma-12b-target-100"),
    RowSpec(5, "llama", "Oracle", "-", "Simnpo", "aggressive", "unlearning-cleanslate/llama-3_1-8b-simnpo-baseline-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(6, "llama", "BM25", "Midtraining", "simnpo", "gentle", "unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-bm25-10b-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(7, "qwen3 8b", "BM25", "Midtraining", "simnpo", "gentle", "unlearning-cleanslate/qwen3-8b-simnpo-gentle-bm25-10b-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
    RowSpec(8, "llama", "IGM", "Midtraining", "simnpo", "gentle", "unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-igm-10b-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(9, "qwen3 8b", "IGM", "Midtraining", "simnpo", "gentle", "unlearning-cleanslate/qwen3-8b-simnpo-gentle-igm-10b-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
    RowSpec(10, "llama", "BM25", "Pretraining", "simnpo", "gentle", "unlearning-cleanslate/llama-3_1-8b-simnpo-gentle-bm25-6t-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(11, "qwen3 8b", "BM25", "Pretraining", "simnpo", "gentle", "unlearning-cleanslate/qwen3-8b-simnpo-gentle-bm25-6t-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
    RowSpec(12, "llama", "Oracle", "-", "undial", "-", "unlearning-cleanslate/llama-3_1-8b-undial-baseline-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(13, "qwen3 8b", "Oracle", "-", "undial", "-", "unlearning-cleanslate/qwen3-8b-undial-baseline-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
    RowSpec(14, "llama", "Oracle", "-", "RMU", "-", "unlearning-cleanslate/llama-3_1-8b-rmu-baseline-target-100", "unlearning-cleanslate/fsid-curated-llama-8b-target-100"),
    RowSpec(15, "qwen3 8b", "Oracle", "-", "RMU", "-", "unlearning-cleanslate/qwen3-8b-rmu-baseline-target-100", "unlearning-cleanslate/fsid-curated-qwen3-8b-target-100"),
)


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)(?:/|$)")
TRAINING_STATE_PATTERNS = (
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
)
BAD_TOKENIZER_CLASS = "TokenizersBackend"
GOOD_TOKENIZER_CLASS = "PreTrainedTokenizerFast"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_gemma(row: RowSpec) -> bool:
    return row.model.lower().startswith("gemma")


def _row_prefix(row: RowSpec) -> str:
    return f"{row.row_id:02d}-{row.checkpoint_repo.split('/')[-1]}"


def _verify_local_model(local_dir: Path) -> tuple[bool, dict[str, Any]]:
    config = local_dir / "config.json"
    weight_index = local_dir / "model.safetensors.index.json"
    safetensors = list(local_dir.glob("*.safetensors"))
    pytorch_bins = list(local_dir.glob("pytorch_model*.bin"))
    tokenizer_files = [
        local_dir / "tokenizer.json",
        local_dir / "tokenizer.model",
        local_dir / "tokenizer_config.json",
    ]
    details: dict[str, Any] = {
        "has_config": config.exists(),
        "has_weight_index": weight_index.exists(),
        "safetensors_count": len(safetensors),
        "pytorch_bin_count": len(pytorch_bins),
        "has_tokenizer": any(p.exists() for p in tokenizer_files),
    }
    ok = bool(
        details["has_config"]
        and (details["has_weight_index"] or details["safetensors_count"] or details["pytorch_bin_count"])
        and details["has_tokenizer"]
    )
    return ok, details


def _patch_tokenizer_config(local_dir: Path) -> list[str]:
    """Repair known training-time tokenizer_config issues in-place.

    The materialized checkpoint dirs are symlink forests into the HF cache. If a
    tokenizer config needs repair, unlink the symlink and write a real JSON file
    in the local materialized dir so the shared cache remains untouched.
    """
    cfg_path = local_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return []

    with cfg_path.open() as f:
        cfg = json.load(f)

    changes: list[str] = []
    if cfg.get("tokenizer_class") == BAD_TOKENIZER_CLASS:
        cfg["tokenizer_class"] = GOOD_TOKENIZER_CLASS
        changes.append(f"tokenizer_class {BAD_TOKENIZER_CLASS}->{GOOD_TOKENIZER_CLASS}")
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg["extra_special_tokens"] = {}
        changes.append("extra_special_tokens list->{}")
    for key in ("backend", "is_local", "local_files_only"):
        if key in cfg:
            cfg.pop(key, None)
            changes.append(f"drop {key}")

    if not changes:
        return []
    if cfg_path.is_symlink() or cfg_path.exists():
        cfg_path.unlink()
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    return changes


def _symlink_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            continue
        os.symlink(path.resolve(), target)


def _compare_reuse_manifest(local_manifest: Path, expected: dict[str, Any]) -> None:
    with local_manifest.open() as f:
        existing = json.load(f)
    keys = ("row_id", "checkpoint_repo", "checkpoint", "checkpoint_number", "repo_revision")
    mismatches = [
        f"{key}: existing={existing.get(key)!r} expected={expected.get(key)!r}"
        for key in keys
        if existing.get(key) != expected.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            f"{local_manifest} does not match requested row:\n  " + "\n  ".join(mismatches)
        )


def _repo_revision(api: HfApi, repo_id: str) -> str:
    info = api.model_info(repo_id=repo_id)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"Could not resolve commit SHA for {repo_id}")
    return str(sha)


def _latest_checkpoint(files: list[str], repo_id: str) -> tuple[str, int]:
    nums = sorted(
        {
            int(match.group(1))
            for name in files
            if (match := CHECKPOINT_RE.match(name))
        }
    )
    if not nums:
        raise RuntimeError(f"No checkpoint-N/ subfolders found in {repo_id}")
    n = nums[-1]
    return f"checkpoint-{n}", n


def _prepare_row(api: HfApi, row: RowSpec, cache_root: Path, hf_org: str) -> dict[str, Any]:
    print(f"\n=== row {row.row_id:02d}: {row.checkpoint_repo} ===", flush=True)
    revision = _repo_revision(api, row.checkpoint_repo)
    files = api.list_repo_files(row.checkpoint_repo, repo_type="model", revision=revision)
    checkpoint, checkpoint_number = _latest_checkpoint(files, row.checkpoint_repo)
    clean = f"{_row_prefix(row)}-{checkpoint}"
    local_dir = cache_root / clean
    local_manifest = local_dir / "manifest.json"
    row_class = "12b" if _is_gemma(row) else "8b"
    eval_config = (
        "scripts/final_final/slurm/config_slurm_eval_gemma.yaml"
        if _is_gemma(row)
        else "scripts/final_final/slurm/config_slurm_eval.yaml"
    )
    slurm_mem = "96G" if _is_gemma(row) else "64G"
    slurm_time = "14:00:00" if _is_gemma(row) else "12:00:00"
    validate_gpu_mem = 0.85 if _is_gemma(row) else 0.90
    expected = {
        **asdict(row),
        "row_id_padded": f"{row.row_id:02d}",
        "repo_basename": row.checkpoint_repo.split("/")[-1],
        "checkpoint": checkpoint,
        "checkpoint_number": checkpoint_number,
        "repo_revision": revision,
        "local_dir": str(local_dir),
    }

    if local_manifest.exists():
        _compare_reuse_manifest(local_manifest, expected)
        print(f"  reuse: {local_dir}", flush=True)
    else:
        if local_dir.exists() and any(local_dir.iterdir()):
            raise RuntimeError(f"{local_dir} exists without manifest.json; refusing to reuse")
        snapshot_path = Path(
            snapshot_download(
                repo_id=row.checkpoint_repo,
                repo_type="model",
                revision=revision,
                allow_patterns=[f"{checkpoint}/**"],
                ignore_patterns=[
                    f"{checkpoint}/{pattern}" for pattern in TRAINING_STATE_PATTERNS
                ],
            )
        )
        src = snapshot_path / checkpoint
        if not src.exists():
            raise RuntimeError(f"Downloaded snapshot lacks expected subdir: {src}")
        local_dir.mkdir(parents=True, exist_ok=True)
        _symlink_tree(src, local_dir)
        expected["snapshot_path"] = str(snapshot_path)
        expected["materialized_at"] = _utc_now()
        expected["materialization"] = "symlink"
        expected["ignored_training_state_patterns"] = list(TRAINING_STATE_PATTERNS)
        with local_manifest.open("w") as f:
            json.dump(expected, f, indent=2, sort_keys=True)
        print(f"  materialized: {local_dir}", flush=True)

    with local_manifest.open() as f:
        manifest = json.load(f)
    tokenizer_config_patches = _patch_tokenizer_config(local_dir)
    ok, verify = _verify_local_model(local_dir)
    if not ok:
        raise RuntimeError(f"{local_dir} failed model-file verification: {verify}")
    manifest.update(
        {
            "verified": True,
            "verified_at": _utc_now(),
            "verification": verify,
            "tokenizer_config_patches": tokenizer_config_patches,
            "row_class": row_class,
            "eval_config": eval_config,
            "slurm_mem": slurm_mem,
            "slurm_time": slurm_time,
            "validate_gpu_mem": validate_gpu_mem,
            "eval_hf_repo": f"{hf_org}/eval-{clean}",
            "generations_hf_repo": f"{hf_org}/generations-{clean}",
        }
    )
    with local_manifest.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"  latest: {checkpoint} @ {revision[:12]}", flush=True)
    print(f"  verify: ok {verify}", flush=True)
    if tokenizer_config_patches:
        print(f"  tokenizer_config: {', '.join(tokenizer_config_patches)}", flush=True)
    return manifest


def _prepare_row_process(row: RowSpec, cache_root: Path, hf_org: str) -> dict[str, Any]:
    return _prepare_row(HfApi(), row, cache_root, hf_org)


def _write_aggregate(rows: list[dict[str, Any]], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _utc_now(),
        "row_count": len(rows),
        "rows": rows,
    }
    json_path = results_dir / "post_eval_manifest.json"
    tsv_path = results_dir / "post_eval_manifest.tsv"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    fieldnames = [
        "row_id",
        "model",
        "curation",
        "data_source",
        "unlearning",
        "setting",
        "checkpoint_repo",
        "hf_datapath",
        "checkpoint",
        "checkpoint_number",
        "repo_revision",
        "local_dir",
        "eval_config",
        "slurm_mem",
        "slurm_time",
        "validate_gpu_mem",
        "eval_hf_repo",
        "generations_hf_repo",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {json_path}")
    print(f"Wrote {tsv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--hf-org", default=DEFAULT_HF_ORG)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of row-prep worker processes. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete cache dirs for the 15 manifest rows before preparing them.",
    )
    parser.add_argument(
        "--row-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional row id whitelist. Defaults to all configured rows.",
    )
    parser.add_argument(
        "--exclude-row",
        action="append",
        type=int,
        default=[],
        help="Row id to exclude. Can be passed more than once.",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Fail unless this many rows are prepared after filtering.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        print("--workers must be >= 1", file=sys.stderr)
        return 2
    selected_rows = list(ROWS)
    if args.row_ids is not None:
        wanted = set(args.row_ids)
        known = {row.row_id for row in ROWS}
        unknown = sorted(wanted - known)
        if unknown:
            print(f"Unknown row ids: {unknown}", file=sys.stderr)
            return 2
        selected_rows = [row for row in selected_rows if row.row_id in wanted]
    excluded = set(args.exclude_row)
    selected_rows = [row for row in selected_rows if row.row_id not in excluded]
    if not selected_rows:
        print("No rows selected after filtering", file=sys.stderr)
        return 2

    if args.expected_rows is not None and len(selected_rows) != args.expected_rows:
        print(
            f"Expected {args.expected_rows} selected rows, got {len(selected_rows)}: "
            f"{[row.row_id for row in selected_rows]}",
            file=sys.stderr,
        )
        return 1

    args.cache_root.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for row in selected_rows:
            prefix = f"{_row_prefix(row)}-checkpoint-"
            for path in args.cache_root.glob(f"{prefix}*"):
                if path.is_dir():
                    print(f"remove {path}")
                    shutil.rmtree(path)

    prepared: list[dict[str, Any]] = []
    failures: list[str] = []
    max_workers = min(args.workers, len(selected_rows))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prepare_row_process, row, args.cache_root, args.hf_org): row
            for row in selected_rows
        }
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                prepared.append(future.result())
            except Exception as exc:  # noqa: BLE001 - collect all row failures, then fail loudly.
                msg = f"row {row.row_id:02d} {row.checkpoint_repo}: {exc}"
                print(f"ERROR: {msg}", file=sys.stderr, flush=True)
                failures.append(msg)

    if failures:
        print("\nPreparation failed; do not submit Slurm jobs.", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    if len(prepared) != len(selected_rows):
        print(f"Expected {len(selected_rows)} rows, prepared {len(prepared)}", file=sys.stderr)
        return 1
    prepared.sort(key=lambda row: int(row["row_id"]))
    expected_ids = [row.row_id for row in selected_rows]
    actual_ids = [int(row["row_id"]) for row in prepared]
    if actual_ids != expected_ids:
        print(f"Expected row ids {expected_ids}, prepared {actual_ids}", file=sys.stderr)
        return 1
    _write_aggregate(prepared, args.results_dir)
    print("\nAll rows verified. Review the TSV manifest before submitting Slurm jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
