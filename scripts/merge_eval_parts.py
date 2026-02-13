"""Merge parallel eval parts and push to HuggingFace."""

import argparse
import logging
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_parts(
    input_dir: str,
    hf_config: str,
    hf_push: str | None = None,
    output_file: str | None = None,
) -> Dataset:
    """
    Merge all part_* subdirectories into a single dataset.

    Args:
        input_dir: Base output directory (e.g., results/eval)
        hf_config: Config name (e.g., direct__first_line)
        hf_push: Optional HuggingFace repo to push to
        output_file: Optional path to save merged parquet

    Returns:
        Merged dataset
    """
    config_dir = Path(input_dir) / hf_config

    # Find all part directories
    part_dirs = sorted(config_dir.glob("part_*"))
    if not part_dirs:
        raise ValueError(f"No part_* directories found in {config_dir}")

    logger.info(f"Found {len(part_dirs)} parts to merge")

    # Load all parquet files
    datasets = []
    for part_dir in part_dirs:
        parquet_path = part_dir / "results.parquet"
        if not parquet_path.exists():
            logger.warning(f"Skipping {part_dir} - no results.parquet found")
            continue

        logger.info(f"Loading {parquet_path}")
        ds = load_dataset("parquet", data_files=str(parquet_path), split="train")
        datasets.append(ds)
        logger.info(f"  Loaded {len(ds)} records")

    if not datasets:
        raise ValueError("No valid parquet files found")

    # Concatenate all datasets
    logger.info("Merging datasets...")
    merged = concatenate_datasets(datasets)
    logger.info(f"Merged dataset has {len(merged)} total records")

    # Save merged parquet
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = config_dir / "results_merged.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(output_path))
    logger.info(f"Saved merged results to {output_path}")

    # Push to HuggingFace
    if hf_push:
        logger.info(f"Pushing to HuggingFace: {hf_push} (config: {hf_config})")
        merged.push_to_hub(
            hf_push,
            config_name=hf_config,
            split="train",
        )
        logger.info(f"Pushed to https://huggingface.co/datasets/{hf_push}")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge parallel eval parts")
    parser.add_argument(
        "-i",
        "--input-dir",
        default="results/eval",
        help="Base output directory (default: results/eval)",
    )
    parser.add_argument(
        "-c",
        "--hf-config",
        required=True,
        help="Config name (e.g., direct__first_line)",
    )
    parser.add_argument(
        "--hf-push",
        help="HuggingFace repo to push merged results",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for merged parquet (default: {input_dir}/{config}/results_merged.parquet)",
    )

    args = parser.parse_args()

    merge_parts(
        input_dir=args.input_dir,
        hf_config=args.hf_config,
        hf_push=args.hf_push,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
