"""Sliding window memorization evaluation via p_z.

Windows are defined in character space then tokenised, avoiding multi-byte
corruption from token-space splitting.

References:
    Hayes et al. "Measuring memorization in language models"
    Cooper et al. "Extracting memorized pieces of copyrighted books"
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from cleanslate.backends.base import InferenceBackend, ProbabilityResult
from cleanslate.backends.vllm_backend import VLLMBackend
from cleanslate.config import load_config, get_eval_config, get_vllm_config
from cleanslate.metrics.probability import compute_extraction_probability

logger = logging.getLogger(__name__)


@dataclass
class MemorizationResult:
    """Per-window evaluation result."""

    sample_id: str
    sample_name: str
    window_idx: int = 0
    seed: str = ""
    target: str = ""
    window_start_char: int = 0
    window_end_char: int = 0
    p_z: float = 0.0
    log_prob: float = 0.0
    is_memorized: bool = False
    num_target_tokens: int = 0
    target_log_probs: list[float] = field(default_factory=list)
    target_ranks: list[int] = field(default_factory=list)
    compute_time_ms: float = 0.0
    row_timestamp: str = ""
    content_id: str | None = None
    content_title: str | None = None
    content_creators: list[str] = field(default_factory=list)
    content_year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlidingWindowResult:
    """Aggregate result for one text."""

    text_name: str
    model_name: str
    prefix_len: int
    suffix_len: int
    stride: int
    tau_min: float
    windows: list[MemorizationResult]
    total_windows: int = 0
    memorized_windows: int = 0
    coverage: float = 0.0
    max_p_z: float = 0.0
    mean_p_z: float = 0.0
    median_p_z: float = 0.0
    total_time_ms: float = 0.0


@dataclass
class ExperimentSummary:
    """Summary statistics across samples."""

    experiment_id: str
    timestamp: str
    model_name: str
    dataset_name: str
    num_samples: int
    memorized_count: int
    memorized_fraction: float
    p_z_mean: float
    p_z_std: float
    p_z_min: float
    p_z_max: float
    p_z_median: float
    total_seconds: float
    samples_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Window:
    prefix: str
    suffix: str
    start_char: int
    end_char: int


def _generate_windows(text: str, prefix_len: int, suffix_len: int, stride: int) -> list[_Window]:
    """Slice text into overlapping (prefix, suffix) pairs in char space."""
    total = prefix_len + suffix_len
    if len(text) < total:
        return []
    return [
        _Window(
            prefix=text[s : s + prefix_len],
            suffix=text[s + prefix_len : s + total],
            start_char=s,
            end_char=s + total,
        )
        for s in range(0, len(text) - total + 1, stride)
    ]


def _coverage(windows: list[MemorizationResult], text_len: int) -> float:
    if not windows or text_len == 0:
        return 0.0
    covered = bytearray(text_len)
    for w in windows:
        if w.is_memorized:
            for i in range(w.window_start_char, min(w.window_end_char, text_len)):
                covered[i] = 1
    return sum(covered) / text_len


class MemorizationEvaluator:
    """Sliding window p_z evaluator."""

    def __init__(
        self,
        model: str,
        config_path: str | Path | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.model_name = model
        self._cfg = load_config(config_path)
        self._backend = backend

    @property
    def eval_cfg(self) -> dict[str, Any]:
        return get_eval_config(self._cfg)

    def _get_backend(self) -> InferenceBackend:
        if self._backend is None:
            self._backend = VLLMBackend(self.model_name, get_vllm_config(self._cfg))
        return self._backend

    @property
    def tokenizer(self) -> Any:
        return self._get_backend().tokenizer

    def sliding_window(self, text: str, name: str = "text") -> SlidingWindowResult:
        return self.sliding_window_batch([text], [name])[0]

    def sliding_window_batch(
        self,
        texts: list[str],
        names: list[str] | None = None,
        batch_size: int = 256,
    ) -> list[SlidingWindowResult]:
        if not texts:
            return []

        names = names or [f"text_{i}" for i in range(len(texts))]
        backend = self._get_backend()
        ec = self.eval_cfg
        prefix_len = ec.get("window_prefix_len", 100)
        suffix_len = ec.get("window_suffix_len", 100)
        stride = ec.get("window_stride", 10)
        tau_min = ec.get("tau_min", 0.001)
        start = time.time()

        per_text = [_generate_windows(t, prefix_len, suffix_len, stride) for t in texts]

        flat: list[tuple[int, int, _Window]] = [
            (ti, wi, w) for ti, ws in enumerate(per_text) for wi, w in enumerate(ws)
        ]

        pairs = [(w.prefix, w.suffix) for _, _, w in flat]
        probs: list[ProbabilityResult] = []
        from tqdm import tqdm
        pbar = tqdm(total=len(pairs), desc="Windows", unit="win")
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            probs.extend(backend.compute_batch(batch))
            pbar.update(len(batch))
        pbar.close()

        grouped: dict[int, list[MemorizationResult]] = {i: [] for i in range(len(texts))}
        now = datetime.now().isoformat()

        for (ti, wi, win), prob in zip(flat, probs):
            ext = compute_extraction_probability(prob.token_log_probs)
            grouped[ti].append(
                MemorizationResult(
                    sample_id=f"{names[ti]}_w{wi}",
                    sample_name=names[ti],
                    window_idx=wi,
                    seed=win.prefix,
                    target=win.suffix,
                    window_start_char=win.start_char,
                    window_end_char=win.end_char,
                    p_z=ext.prob,
                    log_prob=ext.log_prob,
                    is_memorized=(ext.prob >= tau_min),
                    num_target_tokens=ext.num_tokens,
                    target_log_probs=prob.token_log_probs,
                    target_ranks=prob.token_ranks,
                    row_timestamp=now,
                )
            )

        results: list[SlidingWindowResult] = []
        for ti, text in enumerate(texts):
            wrs = grouped[ti]
            pzs = [w.p_z for w in wrs]
            results.append(
                SlidingWindowResult(
                    text_name=names[ti],
                    model_name=self.model_name,
                    prefix_len=prefix_len,
                    suffix_len=suffix_len,
                    stride=stride,
                    tau_min=tau_min,
                    windows=wrs,
                    total_windows=len(wrs),
                    memorized_windows=sum(1 for w in wrs if w.is_memorized),
                    coverage=_coverage(wrs, len(text)),
                    max_p_z=max(pzs) if pzs else 0.0,
                    mean_p_z=statistics.mean(pzs) if pzs else 0.0,
                    median_p_z=statistics.median(pzs) if pzs else 0.0,
                )
            )

        elapsed_ms = (time.time() - start) * 1000
        for r in results:
            r.total_time_ms = elapsed_ms / len(results)
        return results

    def compute_summary(
        self, results: list[MemorizationResult], dataset_name: str = "dataset"
    ) -> ExperimentSummary:
        pzs = [r.p_z for r in results]
        mem = sum(1 for r in results if r.is_memorized)
        ms = sum(r.compute_time_ms for r in results)
        return ExperimentSummary(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            dataset_name=dataset_name,
            num_samples=len(results),
            memorized_count=mem,
            memorized_fraction=mem / len(results) if results else 0.0,
            p_z_mean=statistics.mean(pzs),
            p_z_std=statistics.stdev(pzs) if len(pzs) > 1 else 0.0,
            p_z_min=min(pzs),
            p_z_max=max(pzs),
            p_z_median=statistics.median(pzs),
            total_seconds=ms / 1000,
            samples_per_second=len(results) / (ms / 1000) if ms > 0 else 0,
        )


def save_results(
    results: list[MemorizationResult],
    output_dir: str | Path,
    eval_config: dict[str, Any],
    model_name: str,
    dataset_name: str,
    summary: ExperimentSummary | None = None,
) -> None:
    """Save results to parquet + JSON metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "config": eval_config,
                "num_samples": len(results),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    pq.write_table(
        pa.Table.from_pylist([r.to_dict() for r in results]),
        out / "results.parquet",
    )

    if summary:
        with open(out / "summary.json", "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
