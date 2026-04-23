"""Shared retain-set loaders for FSID curation rules."""

import logging
import random
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def load_wikitext_retain_rows(
    *,
    retain_dataset: str,
    retain_config: str,
    retain_split: str,
    retain_samples: int | None,
    seed: int,
    rule_name: str,
) -> list[dict[str, Any]]:
    """Load WikiText rows for the FSID retain set.

    Filters blank / short headings (<=100 chars), optionally subsamples to
    retain_samples, and stamps each row with rule_name.
    """
    logger.info("Loading retain data from %s/%s split=%s", retain_dataset, retain_config, retain_split)
    wiki = load_dataset(retain_dataset, retain_config, split=retain_split)
    texts = [row["text"] for row in wiki if len(row["text"].strip()) > 100]
    if retain_samples is not None:
        rng = random.Random(seed)
        texts = rng.sample(texts, min(retain_samples, len(texts)))
    rows = [{"text": text, "rule_name": rule_name} for text in texts]
    logger.info("Built %d retain rows from %s", len(rows), retain_dataset)
    return rows
