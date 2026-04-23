"""Load CleanSlate queries as a Polars DataFrame."""

from pathlib import Path

import polars as pl
from datasets import load_dataset

from curation.retrieval.schema import Cfg


def load_queries(
    cfg: Cfg,
    limit: int | None = None,
    content_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Return DataFrame[content_id, content_title, reference_target]."""
    if cfg.dataset_path:
        dataset_path = Path(cfg.dataset_path)
        if dataset_path.suffix == ".parquet":
            df = pl.read_parquet(dataset_path)
        else:
            df = pl.read_ndjson(dataset_path)
    else:
        dataset = load_dataset(
            cfg.dataset_id,
            cfg.dataset_config,
            split=cfg.dataset_split,
            cache_dir=str(cfg.curation_path / "temp" / "hf_cache"),
        )
        df = pl.from_arrow(dataset.data.table)

    df = df.select("content_id", "content_title", "reference_target")
    if content_ids:
        df = df.filter(pl.col("content_id").is_in(content_ids))
    if limit is not None:
        df = df.head(limit)
    return df
