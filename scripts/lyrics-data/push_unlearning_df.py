"""Push local FSID request/set artifacts to Hugging Face dataset repos."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import HfApi


ROOT = Path("/lfs/skampere2/0/animjha/CleanSlate")

MODEL_ARTIFACTS: dict[str, dict[str, object]] = {
    "olmo7b": {
        "repo_id": "unlearning-cleanslate/olmo7B_pool_mem_frac_gt_0.10",
        "artifacts": {
            "pool_requests": ROOT / "data/fsid/olmo7b/pool.parquet",
            "forget_requests": ROOT / "data/fsid/olmo7b/forget.parquet",
            "retain_requests": ROOT / "data/fsid/olmo7b/retain.parquet",
            "forget_set": ROOT / "data/fsid/curated/baseline/olmo7b/forget_set.parquet",
            "retain_set": ROOT / "data/fsid/curated/baseline/olmo7b/retain_set.parquet",
        },
        "metadata_path": ROOT / "data/fsid/curated/baseline/olmo7b/metadata.json",
    },
    "olmo32b": {
        "repo_id": "unlearning-cleanslate/olmo32B_pool_mem_frac_gt_0.10",
        "artifacts": {
            "pool_requests": ROOT / "data/fsid/olmo32b/pool.parquet",
            "forget_requests": ROOT / "data/fsid/olmo32b/forget.parquet",
            "retain_requests": ROOT / "data/fsid/olmo32b/retain.parquet",
            "forget_set": ROOT
            / "data/fsid/curated/baseline/olmo32b/forget_set.parquet",
            "retain_set": ROOT
            / "data/fsid/curated/baseline/olmo32b/retain_set.parquet",
        },
        "metadata_path": ROOT / "data/fsid/curated/baseline/olmo32b/metadata.json",
    },
}


def _load_local_parquet(path: Path):
    table = pq.read_table(path)
    return Dataset(table.replace_schema_metadata(None))


def push_model_artifacts(
    *,
    model_name: str,
    repo_id: str,
    artifact_paths: dict[str, Path],
    metadata_path: Path,
    dry_run: bool,
) -> None:
    print(f"\n[{model_name}] repo={repo_id}")

    for artifact_name, local_path in artifact_paths.items():
        if not local_path.exists():
            raise FileNotFoundError(
                f"Missing required artifact for {model_name}: {local_path}"
            )

        dataset = _load_local_parquet(local_path)
        print(
            f"  config={artifact_name} split=train rows={len(dataset)} path={local_path}"
        )
        if dry_run:
            continue

        dataset.push_to_hub(
            repo_id,
            config_name=artifact_name,
            split="train",
            private=False,
        )

    if metadata_path.exists():
        print(f"  metadata={metadata_path}")
        if not dry_run:
            HfApi().upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
    else:
        print(f"  metadata missing, skipping: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push local FSID request/set artifacts to Hugging Face."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_ARTIFACTS),
        action="append",
        dest="models",
        help="Model bundle(s) to push. Default: all configured bundles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned pushes without uploading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = args.models or sorted(MODEL_ARTIFACTS)

    for model_name in selected_models:
        config = MODEL_ARTIFACTS[model_name]
        push_model_artifacts(
            model_name=model_name,
            repo_id=str(config["repo_id"]),
            artifact_paths=dict(config["artifacts"]),
            metadata_path=Path(config["metadata_path"]),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
