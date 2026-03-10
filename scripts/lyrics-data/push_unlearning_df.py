"""Push local FSID artifacts from final pipeline to Hugging Face dataset repos.

All artifacts use threshold=0.05 and SimNPO objective.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi


ROOT = Path("/data/animesh_jha/CleanSlate")

ALL_ALIASES = [
    "olmo-7b",
    "llama-8b",
    "qwen-8b",
    "qwen-8b-base",
    "trinity-nano",
]

HF_ORG = "unlearning-cleanslate"


def _model_paths(alias: str) -> dict[str, Path]:
    """Derive all artifact paths for a model alias from the final pipeline layout."""
    return {
        # Pool artifacts (threshold=0.05)
        "pool": ROOT / f"data/final/{alias}/pool/pool.parquet",
        "forget_pool": ROOT / f"data/final/{alias}/pool/forget.parquet",
        "retain_pool": ROOT / f"data/final/{alias}/pool/retain.parquet",
        # Curated sets
        "forget_set": ROOT / f"data/final/curated/{alias}/baseline/forget_set.parquet",
        "retain_set": ROOT / f"data/final/curated/{alias}/baseline/retain_set.parquet",
        "requests": ROOT / f"data/final/curated/{alias}/baseline/requests.parquet",
        # Post-eval results (SimNPO)
        "post_eval_simnpo": ROOT / f"results/final/post_eval/{alias}/simnpo/default/results.parquet",
    }


def _json_paths(alias: str) -> dict[str, Path]:
    """Derive JSON artifact paths (metadata, scores, validation)."""
    score = ROOT / f"results/final/score/{alias}"
    val = ROOT / f"results/final/validation/{alias}"
    curated = ROOT / f"data/final/curated/{alias}/baseline"
    return {
        "metadata": curated / "metadata.json",
        # Scores (SimNPO only)
        "simnpo_score": score / "simnpo_score.json",
        "simnpo_qa_score": score / "simnpo_qa_score.json",
        # Validation
        "pre_gsm8k": val / "pre_gsm8k.json",
        "pre_qa": val / "pre_qa.json",
        "post_simnpo_gsm8k": val / "post_simnpo_gsm8k.json",
        "post_simnpo_qa": val / "post_simnpo_qa.json",
    }


def _load_local_parquet(path: Path) -> Dataset:
    table = pq.read_table(path)
    return Dataset(table.replace_schema_metadata(None))


def push_model_artifacts(
    alias: str,
    *,
    dry_run: bool,
    token: str | None = None,
) -> None:
    repo_id = f"{HF_ORG}/fsid_{alias}"
    parquet_paths = _model_paths(alias)
    json_paths = _json_paths(alias)

    print(f"\n[{alias}] repo={repo_id}")

    # Push parquet artifacts as HF dataset configs
    for config_name, local_path in parquet_paths.items():
        if not local_path.exists():
            print(f"  SKIP {config_name}: {local_path} not found")
            continue

        dataset = _load_local_parquet(local_path)
        print(f"  config={config_name}  rows={len(dataset)}  path={local_path}")
        if dry_run:
            continue

        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            split="train",
            private=False,
            token=token,
        )

    # Upload JSON files (scores, validation, metadata)
    api = HfApi(token=token)
    for name, local_path in json_paths.items():
        if not local_path.exists():
            print(f"  SKIP {name}: {local_path} not found")
            continue

        print(f"  json={name}  path={local_path}")
        if dry_run:
            continue

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{name}.json",
            repo_id=repo_id,
            repo_type="dataset",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push final pipeline FSID artifacts to Hugging Face."
    )
    parser.add_argument(
        "--model",
        choices=ALL_ALIASES,
        action="append",
        dest="models",
        help="Model alias(es) to push. Default: all 5 models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pushes without uploading anything.",
    )
    return parser.parse_args()


def main() -> None:
    import os

    args = parse_args()
    selected = args.models or ALL_ALIASES
    token = os.environ.get("HF_TOKEN")

    for alias in selected:
        push_model_artifacts(alias, dry_run=args.dry_run, token=token)


if __name__ == "__main__":
    main()
