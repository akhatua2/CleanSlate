"""Evaluation harness for generated outputs.

Computes metrics (BLEU, ROUGE, extraction probability, perplexity) comparing
generated text against reference targets.
Saves results as JSON/Parquet for analysis.

NOTE: Probability metrics are computed from pre-existing token_log_probs in the
dataset - no model loading required. The generation step already captured all
the probability data we need.

OPTIMIZATIONS:
- Probability metrics: Vectorized with polars (no Python loop)
- BLEU/ROUGE: Parallelized across CPU cores with multiprocessing
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from multiprocessing import Pool, cpu_count
from functools import partial

import polars as pl
from tqdm import tqdm

from cleanslate.metrics.bleu import BLEU1, BLEU2, modified_precision, brevity_penalty
from cleanslate.metrics.rouge import ROUGE1, ROUGE2, ROUGEL, lcs_length
from cleanslate.metrics.probability import compute_extraction_probability


@dataclass
class EvalRow:
    """Single evaluation result - one row per sample with all metrics as columns."""

    sample_name: str
    prompt_template: str
    seed_mode: str
    # Similarity metrics
    bleu1: float
    bleu2: float
    rouge1: float
    rouge2: float
    rougel: float
    # Probability metrics (from existing token_log_probs)
    log_prob: float
    avg_log_prob: float
    perplexity: float
    extraction_prob: float
    num_tokens: int
    # Text data
    generated_text: str
    reference_target: str


# --- Standalone functions for multiprocessing (must be picklable) ---

def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.lower().split()


def _compute_bleu1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute BLEU-1 score."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    p = modified_precision(pred_tokens, ref_tokens, 1)
    if p == 0:
        return 0.0
    bp = brevity_penalty(pred_tokens, ref_tokens)
    return bp * p


def _compute_bleu2(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute BLEU-2 score."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    p1 = modified_precision(pred_tokens, ref_tokens, 1)
    p2 = modified_precision(pred_tokens, ref_tokens, 2)
    if p1 == 0 or p2 == 0:
        return 0.0
    bp = brevity_penalty(pred_tokens, ref_tokens)
    return bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))


def _compute_rouge1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute ROUGE-1 F1 score."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    from collections import Counter
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(c, ref_counts.get(t, 0)) for t, c in pred_counts.items())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_rouge2(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute ROUGE-2 F1 score."""
    if len(pred_tokens) < 2 or len(ref_tokens) < 2:
        return 0.0
    from collections import Counter
    pred_bigrams = Counter((pred_tokens[i], pred_tokens[i+1]) for i in range(len(pred_tokens)-1))
    ref_bigrams = Counter((ref_tokens[i], ref_tokens[i+1]) for i in range(len(ref_tokens)-1))
    overlap = sum(min(c, ref_bigrams.get(b, 0)) for b, c in pred_bigrams.items())
    precision = overlap / sum(pred_bigrams.values())
    recall = overlap / sum(ref_bigrams.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_rougel(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute ROUGE-L F1 score."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_all_similarity(args: tuple[str, str]) -> dict[str, float]:
    """Compute all similarity metrics for a single (generated, reference) pair."""
    generated, reference = args
    pred_tokens = _tokenize(generated)
    ref_tokens = _tokenize(reference)

    return {
        "bleu1": _compute_bleu1(pred_tokens, ref_tokens),
        "bleu2": _compute_bleu2(pred_tokens, ref_tokens),
        "rouge1": _compute_rouge1(pred_tokens, ref_tokens),
        "rouge2": _compute_rouge2(pred_tokens, ref_tokens),
        "rougel": _compute_rougel(pred_tokens, ref_tokens),
    }


class EvaluationExperiment:
    """Run evaluation metrics on generated outputs.

    Optimized with:
    - Vectorized probability metrics (polars)
    - Parallel BLEU/ROUGE computation (multiprocessing)
    """

    def __init__(self, n_workers: int | None = None):
        """Initialize evaluation experiment.

        Args:
            n_workers: Number of parallel workers for BLEU/ROUGE. Default: cpu_count()
        """
        self.n_workers = n_workers or cpu_count()
        self.available_similarity_metrics = ["bleu1", "bleu2", "rouge1", "rouge2", "rougel"]

    def run(
        self,
        dataset_id: str,
        split: str,
        output_path: str | None = None,
        metrics: list[str] | None = None,
        limit: int | None = None,
    ) -> list[EvalRow]:
        """Evaluate a dataset with optimized parallel processing."""
        from pathlib import Path as PathlibPath

        # --- Load data ---
        if PathlibPath(dataset_id).exists():
            from datasets import load_from_disk
            ds_full = load_from_disk(dataset_id)
            df = pl.DataFrame(list(ds_full[split]))
            if limit:
                df = df.head(limit)
        else:
            df = self._load_from_hub(dataset_id, split, limit)

        print(f"Loaded {len(df)} rows")

        # Filter out empty rows
        df = df.filter(
            (pl.col("generated_text").is_not_null()) &
            (pl.col("generated_text") != "") &
            (pl.col("reference_target").is_not_null()) &
            (pl.col("reference_target") != "")
        )

        # --- Vectorized probability metrics ---
        print("Computing probability metrics (vectorized)...")
        df = self._compute_probability_metrics_vectorized(df)

        # --- Parallel BLEU/ROUGE ---
        print(f"Computing similarity metrics ({self.n_workers} workers)...")
        similarity_scores = self._compute_similarity_parallel(df)

        # Add similarity scores to dataframe
        df = df.with_columns([
            pl.Series("bleu1", [s["bleu1"] for s in similarity_scores]),
            pl.Series("bleu2", [s["bleu2"] for s in similarity_scores]),
            pl.Series("rouge1", [s["rouge1"] for s in similarity_scores]),
            pl.Series("rouge2", [s["rouge2"] for s in similarity_scores]),
            pl.Series("rougel", [s["rougel"] for s in similarity_scores]),
        ])

        # --- Build results ---
        # Parse prompt_template and seed_mode from split name if not in data
        prompt_template_default = ""
        seed_mode_default = ""
        if "__" in split:
            parts = split.split("__")
            prompt_template_default = parts[0]
            seed_mode_default = parts[1] if len(parts) > 1 else ""

        results = []
        for row in df.iter_rows(named=True):
            eval_row = EvalRow(
                sample_name=row.get("sample_name", ""),
                prompt_template=row.get("prompt_template", prompt_template_default),
                seed_mode=row.get("seed_mode", seed_mode_default),
                bleu1=row["bleu1"],
                bleu2=row["bleu2"],
                rouge1=row["rouge1"],
                rouge2=row["rouge2"],
                rougel=row["rougel"],
                log_prob=row["log_prob"],
                avg_log_prob=row["avg_log_prob"],
                perplexity=row["perplexity"],
                extraction_prob=row["extraction_prob"],
                num_tokens=row["num_tokens"],
                generated_text=row["generated_text"][:200],
                reference_target=row["reference_target"][:200],
            )
            results.append(eval_row)

        if output_path:
            self._save_results(results, output_path)

        return results

    def _load_from_hub(self, dataset_id: str, split: str, limit: int | None) -> pl.DataFrame:
        """Load dataset from HuggingFace Hub with parallel download."""
        from huggingface_hub import hf_hub_download, list_repo_files
        from concurrent.futures import ThreadPoolExecutor

        print("Listing files...")
        all_files = list_repo_files(dataset_id, repo_type="dataset")
        parquet_files = [f for f in all_files if f.startswith(f"{split}/") and f.endswith(".parquet")]

        if not parquet_files:
            raise ValueError(f"No parquet files found for {dataset_id}/{split}")

        cache_dir = str(Path(__file__).parent.parent / ".cache" / "huggingface")

        def download_file(filename: str) -> str:
            return hf_hub_download(
                repo_id=dataset_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
            )

        print(f"Downloading {len(parquet_files)} parquet files...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            local_files = list(tqdm(
                executor.map(download_file, parquet_files),
                total=len(parquet_files),
                desc="Downloading"
            ))

        df = pl.concat([pl.read_parquet(f) for f in local_files])

        if limit:
            df = df.head(limit)

        return df

    def _compute_probability_metrics_vectorized(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute probability metrics using polars vectorized operations."""

        # Compute metrics using polars expressions
        df = df.with_columns([
            # num_tokens = length of token_log_probs list
            pl.col("token_log_probs").list.len().alias("num_tokens"),

            # log_prob = sum of token_log_probs
            pl.col("token_log_probs").list.sum().alias("log_prob"),
        ])

        # avg_log_prob and perplexity (need num_tokens > 0)
        df = df.with_columns([
            # avg_log_prob = log_prob / num_tokens
            (pl.col("log_prob") / pl.col("num_tokens")).alias("avg_log_prob"),
        ])

        df = df.with_columns([
            # perplexity = exp(-avg_log_prob)
            (-pl.col("avg_log_prob")).exp().alias("perplexity"),

            # extraction_prob = exp(log_prob) - but clamp to avoid overflow
            pl.col("log_prob").exp().alias("extraction_prob"),
        ])

        # Handle edge cases (null/empty token_log_probs)
        df = df.with_columns([
            pl.col("log_prob").fill_null(0.0),
            pl.col("avg_log_prob").fill_null(0.0),
            pl.col("perplexity").fill_null(1.0),
            pl.col("extraction_prob").fill_null(0.0),
            pl.col("num_tokens").fill_null(0),
        ])

        return df

    def _compute_similarity_parallel(self, df: pl.DataFrame) -> list[dict[str, float]]:
        """Compute BLEU/ROUGE in parallel using multiprocessing."""

        generated_texts = df["generated_text"].to_list()
        reference_texts = df["reference_target"].to_list()
        pairs = list(zip(generated_texts, reference_texts))

        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_all_similarity, pairs, chunksize=100),
                total=len(pairs),
                desc="BLEU/ROUGE"
            ))

        return results

    def _save_results(self, results: list[EvalRow], output_path: str) -> None:
        """Save evaluation results."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
        elif path.suffix == ".parquet":
            data = [asdict(r) for r in results]
            out_df = pl.DataFrame(data)
            out_df.write_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @staticmethod
    def summary(results: list[EvalRow]) -> dict:
        """Compute summary statistics per metric."""
        if not results:
            return {}

        metrics = [
            "bleu1", "bleu2", "rouge1", "rouge2", "rougel",
            "log_prob", "avg_log_prob", "perplexity", "extraction_prob",
        ]
        summary = {}

        for metric_name in metrics:
            scores = [getattr(row, metric_name) for row in results]
            if scores:
                summary[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        return summary
