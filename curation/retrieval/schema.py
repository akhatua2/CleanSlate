"""Config and output schema for retrieval runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
import json
from pathlib import Path

from datasets import Features, Value
import pyarrow as pa

DOLMA_ROOT = Path("/lfs/skampere2/0/shared_hf_cache/dolma3")
CURATION_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = CURATION_ROOT / "vendor" / "infini-gram-mini"


@dataclass(frozen=True)
class Cfg:
    variant: str = "10B"
    method: str = "infini_gram_mini"
    top_k: int = 100
    min_span_chars: int = 20
    segment_chars: int = 2000
    segment_overlap: int = 500
    min_doc_chars: int = 100
    min_segment_chars: int = 200
    categories: str | None = None
    workers: int = 32
    index_workers: int | None = None
    index_mem_gb: int = 256
    num_shards: int | None = None
    dolma_root: str | None = None
    curation_root: str | None = None
    dataset_path: str | None = None
    dataset_id: str = "unlearning-cleanslate/cleanslate_dataset"
    dataset_config: str = "default"
    dataset_split: str = "train"

    @property
    def dolma_path(self) -> Path:
        return Path(self.dolma_root) if self.dolma_root else DOLMA_ROOT

    @property
    def curation_path(self) -> Path:
        return Path(self.curation_root) if self.curation_root else CURATION_ROOT

    @property
    def raw_data(self) -> Path:
        data_dir = self.dolma_path / "raw" / self.variant / "data"
        return data_dir if data_dir.exists() else self.dolma_path / "raw" / self.variant

    @property
    def normalized_num_shards(self) -> int:
        defaults = {"10B": 32, "6T": 512}
        return self.num_shards or defaults.get(self.variant, 64)

    @property
    def normalized_index_workers(self) -> int:
        return self.index_workers or self.workers

    def _hash(self, payload: dict[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return sha1(encoded).hexdigest()[:12]

    @property
    def index_hash(self) -> str:
        return self._hash(
            {
                "variant": self.variant,
                "method": self.method,
                "raw_data": str(self.raw_data),
                "segment_chars": self.segment_chars,
                "segment_overlap": self.segment_overlap,
                "min_doc_chars": self.min_doc_chars,
                "min_segment_chars": self.min_segment_chars,
                "categories": self.categories,
                "num_shards": self.normalized_num_shards,
                "normalization": {
                    "lower": True,
                    "stopwords": "english",
                    "stemming": False,
                    "lemmatization": False,
                },
            }
        )

    @property
    def query_hash(self) -> str:
        dataset_key = self.dataset_path or f"{self.dataset_id}:{self.dataset_config}:{self.dataset_split}"
        return self._hash(
            {
                "index_hash": self.index_hash,
                "dataset": dataset_key,
                "top_k": self.top_k,
                "output_schema_version": 2,
            }
        )

    @property
    def index_dir(self) -> Path:
        return self.curation_path / "indices" / self.variant / self.method / self.index_hash

    @property
    def output_dir(self) -> Path:
        return (
            self.curation_path
            / "outputs"
            / f"{self.method}_cleanslate_dolma3_{self.variant}"
            / self.query_hash
        )

    @property
    def items_dir(self) -> Path:
        return self.output_dir / "items"

    @property
    def matches_path(self) -> Path:
        return self.output_dir / "matches.parquet"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "summary.parquet"

    def raw_shards(self) -> list[Path]:
        paths = sorted(self.raw_data.rglob("*.jsonl.zst"))
        if not self.categories:
            return paths
        cats = {c.strip() for c in self.categories.split(",") if c.strip()}
        return [path for path in paths if any(cat in path.parent.name for cat in cats)]

    def worker_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Span:
    start: int
    end: int
    count: int


MATCH_SCHEMA = pa.schema(
    [
        ("content_id", pa.string()),
        ("content_title", pa.string()),
        ("reference_target", pa.large_string()),
        ("retrieval_method", pa.string()),
        ("corpus_name", pa.string()),
        ("doc_id", pa.string()),
        ("doc_source", pa.string()),
        ("doc_category", pa.string()),
        ("unit_id", pa.string()),
        ("unit_char_start", pa.int64()),
        ("unit_char_end", pa.int64()),
        ("rank", pa.int64()),
        ("score", pa.float64()),
        ("retrieved_text", pa.large_string()),
        ("metadata", pa.large_string()),
    ]
)

HF_FEATURES = Features(
    {
        "content_id": Value("string"),
        "content_title": Value("string"),
        "reference_target": Value("large_string"),
        "retrieval_method": Value("string"),
        "corpus_name": Value("string"),
        "doc_id": Value("string"),
        "doc_source": Value("string"),
        "doc_category": Value("string"),
        "unit_id": Value("string"),
        "unit_char_start": Value("int64"),
        "unit_char_end": Value("int64"),
        "rank": Value("int64"),
        "score": Value("float64"),
        "retrieved_text": Value("large_string"),
        "metadata": Value("large_string"),
    }
)
