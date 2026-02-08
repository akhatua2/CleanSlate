"""Generation-only experiment runner with parquet output.

Runs text generation on datasets and saves all outputs + logits
without computing metrics. Designed for HuggingFace Datasets compatibility.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from .backends.base import InferenceBackend
from .backends.huggingface import HuggingFaceBackend

logger = logging.getLogger(__name__)


@dataclass
class GenerationRow:
    """Single row in the generation output dataset.

    Flat structure for parquet/HuggingFace Datasets compatibility.
    Each row = one generation for one seed.
    """

    # Identifiers
    model_name: str
    dataset_name: str
    sample_name: str
    generation_idx: int  # 0, 1, 2... for multiple generations per seed

    # Input
    seed: str
    reference_target: str  # Expected continuation, may be empty

    # Generation output
    generated_text: str
    generated_token_ids: list[int]
    generated_tokens: list[str]

    # Token-level probability data
    token_log_probs: list[float]
    top_k_tokens: list[list[int]]
    top_k_probs: list[list[float]]

    # Generation parameters
    max_new_tokens: int
    temperature: float
    top_p: float | None
    top_k: int | None
    do_sample: bool

    # Metadata
    timestamp: str
    device: str
    generation_time_ms: float

    def to_dict(self) -> dict:
        """Convert to dict for DataFrame/parquet."""
        return asdict(self)


BackendType = Literal["huggingface", "vllm", "sglang"]


class GenerationExperiment:
    """
    Experiment for running generations and saving outputs.

    Example:
        exp = GenerationExperiment(model="Qwen/Qwen3-0.6B")
        exp.run(
            samples=SHAKESPEARE_SAMPLES,
            dataset_name="shakespeare",
            output_path="generations/shakespeare.parquet",
            num_generations=3,
        )
    """

    def __init__(
        self,
        model: str,
        backend: BackendType = "huggingface",
        device: str | None = None,
        **backend_kwargs,
    ):
        self.model_name = model
        self._backend_type = backend
        self._device = device
        self._backend_kwargs = backend_kwargs
        self._backend: InferenceBackend | None = None

    def _get_backend(self) -> InferenceBackend:
        """Lazy-load the model."""
        if self._backend is None:
            if self._backend_type == "huggingface":
                self._backend = HuggingFaceBackend(self.model_name, device=self._device)
            elif self._backend_type == "vllm":
                from .backends.vllm_backend import VLLMBackend

                self._backend = VLLMBackend(self.model_name)
            elif self._backend_type == "sglang":
                from .backends.sglang_backend import SGLangBackend

                self._backend = SGLangBackend(self.model_name, **self._backend_kwargs)
            else:
                raise ValueError(f"Unknown backend: {self._backend_type}")
        return self._backend

    def run(
        self,
        samples: list[dict],
        dataset_name: str,
        output_path: str | Path,
        num_generations: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        capture_top_k: int = 50,
        batch_size: int = 8,
    ) -> list[GenerationRow]:
        """
        Run generation on all samples and save to parquet.

        Args:
            samples: List of dicts with keys: name, seed, target (optional)
            dataset_name: Name for the dataset (e.g., "shakespeare")
            output_path: Path to save parquet file
            num_generations: Number of generations per seed
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample (vs greedy)
            capture_top_k: Number of top tokens to save per position
            batch_size: Number of sequences to generate in parallel

        Returns:
            List of GenerationRow objects
        """
        backend = self._get_backend()
        rows: list[GenerationRow] = []

        # Build flat list of all generation jobs
        # Each job: (sample_idx, gen_idx, sample_name, seed, reference_target)
        jobs: list[tuple[int, int, str, str, str]] = []
        for sample_idx, sample in enumerate(samples):
            sample_name = sample.get("name", "unnamed")
            seed = sample["seed"]
            reference_target = sample.get("target", "")
            for gen_idx in range(num_generations):
                jobs.append((sample_idx, gen_idx, sample_name, seed, reference_target))

        total_jobs = len(jobs)
        num_batches = (total_jobs + batch_size - 1) // batch_size

        logger.info(
            "Processing %d generations in %d batches (batch_size=%d)",
            total_jobs,
            num_batches,
            batch_size,
        )

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_jobs)
            batch_jobs = jobs[batch_start:batch_end]

            logger.info(
                "[Batch %d/%d] Processing %d generations...",
                batch_idx + 1,
                num_batches,
                len(batch_jobs),
            )

            # Extract seeds for this batch
            batch_seeds = [job[3] for job in batch_jobs]

            # Generate batch
            results = backend.generate_batch(
                seeds=batch_seeds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                capture_top_k=capture_top_k,
            )

            # Create rows for each result
            timestamp = datetime.now().isoformat()
            for job, result in zip(batch_jobs, results):
                sample_idx, gen_idx, sample_name, seed, reference_target = job

                row = GenerationRow(
                    model_name=self.model_name,
                    dataset_name=dataset_name,
                    sample_name=sample_name,
                    generation_idx=gen_idx,
                    seed=seed,
                    reference_target=reference_target,
                    generated_text=result.generated_text,
                    generated_token_ids=result.generated_token_ids,
                    generated_tokens=result.generated_tokens,
                    token_log_probs=result.token_log_probs,
                    top_k_tokens=result.top_k_tokens,
                    top_k_probs=result.top_k_probs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    timestamp=timestamp,
                    device=backend.device,
                    generation_time_ms=result.generation_time_ms,
                )
                rows.append(row)

        # Save outputs (both parquet and json by default)
        path = Path(output_path)
        base = path.parent / path.stem

        # Always save parquet (for HF datasets compatibility)
        self._save_output(rows, f"{base}.parquet", format="parquet")

        # Also save JSON (human readable)
        self._save_output(rows, f"{base}.json", format="json")

        return rows

    def _save_output(
        self,
        rows: list[GenerationRow],
        path: str | Path,
        format: str = "parquet",
    ) -> None:
        """Save rows to file in specified format."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list of dicts
        data = [row.to_dict() for row in rows]

        if format == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif format == "jsonl":
            with open(path, "w") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")
        elif format == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pylist(data)
                pq.write_table(table, path)
            except ImportError:
                import pandas as pd

                df = pd.DataFrame(data)
                df.to_parquet(path, index=False)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'parquet', 'json', or 'jsonl'"
            )

        logger.info("Saved %d generations to %s", len(rows), path)
