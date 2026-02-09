"""Generation-only experiment runner with parquet output.

Runs text generation on datasets and saves all outputs + logits
without computing metrics. Designed for HuggingFace Datasets compatibility.
"""

import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from cleanslate.backends.base import InferenceBackend
from cleanslate.backends import create_backend

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

    # Song metadata (optional)
    song_id: str | None = None
    song_title: str | None = None
    song_artists: list[str] = field(default_factory=list)
    song_year: int | None = None
    wikipedia_link: str | None = None

    # Prompt metadata (optional)
    prompt_id: str | None = None
    prompt_template: str | None = None
    seed_mode: str | None = None
    lyrics_hint: str | None = None

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
        backend: BackendType | InferenceBackend = "huggingface",
        device: str | None = None,
        **backend_kwargs,
    ):
        self.model_name = model
        self._device = device
        self._backend_kwargs = backend_kwargs
        self._backend: InferenceBackend | None = None
        if isinstance(backend, InferenceBackend):
            self._backend = backend
            self._backend_type = "custom"
        else:
            self._backend_type = backend

    def _get_backend(self) -> InferenceBackend:
        """Lazy-load the model."""
        if self._backend is None:
            self._backend = create_backend(
                self._backend_type,
                self.model_name,
                device=self._device,
                **self._backend_kwargs,
            )
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
        hf_repo: str | None = None,
        thinking: bool = False,
    ) -> list[GenerationRow]:
        """
        Run generation on all samples and save to parquet.

        Args:
            samples: List of dicts with keys: name, seed, target (optional).
                     If "messages" is provided, it will be formatted as chat.
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
            thinking: Enable model thinking mode when formatting chat prompts

        Returns:
            List of GenerationRow objects
        """
        backend = self._get_backend()
        rows: list[GenerationRow] = []

        # Build flat list of all generation jobs
        # Each job: (sample_idx, gen_idx, sample_name, seed, reference_target, metadata)
        jobs: list[tuple[int, int, str, str, str, dict]] = []
        for sample_idx, sample in enumerate(samples):
            sample_name = sample.get("sample_name", sample.get("name", "unnamed"))
            if sample.get("messages"):
                if not hasattr(backend, "format_chat"):
                    raise ValueError(
                        f"Backend {self._backend_type} does not support chat formatting"
                    )
                seed = backend.format_chat(
                    sample["messages"],
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            else:
                seed = sample["seed"]
            reference_target = sample.get("target", "")
            metadata = {
                "song_id": sample.get("song_id"),
                "song_title": sample.get("song_title"),
                "song_artists": sample.get("song_artists") or [],
                "song_year": sample.get("song_year"),
                "wikipedia_link": sample.get("wikipedia_link"),
                "prompt_id": sample.get("prompt_id"),
                "prompt_template": sample.get("prompt_template"),
                "seed_mode": sample.get("seed_mode"),
                "lyrics_hint": sample.get("lyrics_hint"),
            }
            for gen_idx in range(num_generations):
                jobs.append(
                    (sample_idx, gen_idx, sample_name, seed, reference_target, metadata)
                )

        total_jobs = len(jobs)
        num_batches = (total_jobs + batch_size - 1) // batch_size

        logger.info(
            "Processing %d generations in %d batches (batch_size=%d)",
            total_jobs,
            num_batches,
            batch_size,
        )

        for batch_idx in tqdm(range(num_batches), desc="Batches", unit="batch"):
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
                (
                    sample_idx,
                    gen_idx,
                    sample_name,
                    seed,
                    reference_target,
                    metadata,
                ) = job

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
                    song_id=metadata.get("song_id"),
                    song_title=metadata.get("song_title"),
                    song_artists=metadata.get("song_artists") or [],
                    song_year=metadata.get("song_year"),
                    wikipedia_link=metadata.get("wikipedia_link"),
                    prompt_id=metadata.get("prompt_id"),
                    prompt_template=metadata.get("prompt_template"),
                    seed_mode=metadata.get("seed_mode"),
                    lyrics_hint=metadata.get("lyrics_hint"),
                )
                rows.append(row)

        # Save parquet locally
        path = Path(output_path)
        base = path.parent / path.stem
        parquet_path = f"{base}.parquet"
        self._save_output(rows, parquet_path, format="parquet")

        # Push to HuggingFace Hub if requested
        if hf_repo:
            self._push_to_hub(rows, hf_repo, dataset_name)

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

    def _push_to_hub(
        self,
        rows: list[GenerationRow],
        hf_repo: str,
        dataset_name: str,
    ) -> None:
        """Push generation rows to HuggingFace Hub as a dataset."""
        from datasets import Dataset

        data = [row.to_dict() for row in rows]
        ds = Dataset.from_list(data)
        ds.push_to_hub(hf_repo, config_name=dataset_name)
        logger.info("Pushed %d rows to %s/%s", len(rows), hf_repo, dataset_name)
