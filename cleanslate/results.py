"""Structured results for experiments."""

import logging
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single (seed, target) sample."""

    name: str
    seed: str
    target: str
    log_likelihood: float
    perplexity: float
    num_tokens: int
    rank_1_fraction: float
    avg_rank: float
    token_log_probs: list[float] = field(default_factory=list)
    token_ranks: list[int] = field(default_factory=list)
    target_tokens: list[str] = field(default_factory=list)
    predicted_tokens: list[str] = field(default_factory=list)
    # N-gram metrics
    bleu_1: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleResult":
        # Handle older results without new fields
        d.setdefault("target_tokens", [])
        d.setdefault("predicted_tokens", [])
        d.setdefault("bleu_1", 0.0)
        d.setdefault("rouge_1", 0.0)
        d.setdefault("rouge_l", 0.0)
        return cls(**d)

    def token_diff_table(self) -> str:
        """Generate a token-by-token diff table."""
        if not self.target_tokens or not self.predicted_tokens:
            return "No token data available."

        lines = []
        lines.append(
            f"{'Pos':>4} │ {'Target':<15} │ {'Predicted':<15} │ {'Rank':>5} │ Match"
        )
        lines.append("─" * 60)

        for i, (tgt, pred, rank) in enumerate(
            zip(self.target_tokens, self.predicted_tokens, self.token_ranks)
        ):
            match = "✓" if rank == 1 else "✗"
            # Escape/repr for visibility of whitespace
            tgt_display = repr(tgt)[1:-1][:15]
            pred_display = repr(pred)[1:-1][:15]
            lines.append(
                f"{i + 1:>4} │ {tgt_display:<15} │ {pred_display:<15} │ {rank:>5} │ {match}"
            )

        return "\n".join(lines)


@dataclass
class ExperimentResults:
    """Results from running an experiment across multiple samples."""

    model_name: str
    experiment_name: str
    timestamp: str
    device: str
    samples: list[SampleResult]

    # Computed on init
    mean_log_likelihood: float = 0.0
    mean_perplexity: float = 0.0
    mean_rank_1_fraction: float = 0.0
    mean_avg_rank: float = 0.0
    mean_bleu_1: float = 0.0
    mean_rouge_1: float = 0.0
    mean_rouge_l: float = 0.0

    def __post_init__(self):
        """Compute aggregate statistics."""
        if self.samples:
            n = len(self.samples)
            self.mean_log_likelihood = sum(s.log_likelihood for s in self.samples) / n
            self.mean_perplexity = sum(s.perplexity for s in self.samples) / n
            self.mean_rank_1_fraction = sum(s.rank_1_fraction for s in self.samples) / n
            self.mean_avg_rank = sum(s.avg_rank for s in self.samples) / n
            self.mean_bleu_1 = sum(s.bleu_1 for s in self.samples) / n
            self.mean_rouge_1 = sum(s.rouge_1 for s in self.samples) / n
            self.mean_rouge_l = sum(s.rouge_l for s in self.samples) / n

    def summary(self, show_tokens: bool = False) -> str:
        """Generate a formatted summary table."""
        lines = []

        # Header
        lines.append(f"Experiment: {self.experiment_name}")
        lines.append(f"Model: {self.model_name}")
        lines.append(f"Device: {self.device}")
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append(f"Samples: {len(self.samples)}")
        lines.append("")

        # Column widths
        name_w = max(30, max((len(s.name[:30]) for s in self.samples), default=10))

        # Table header
        header = f"{'Sample':<{name_w}} │ {'Log-Lik':>9} │ {'PPL':>7} │ {'Rank-1':>6} │ {'AvgRank':>7} │ {'BLEU-1':>6} │ {'ROUGE-1':>7} │ {'ROUGE-L':>7}"
        if show_tokens:
            header += f" │ {'Tok':>4}"
        lines.append("─" * len(header))
        lines.append(header)
        lines.append("─" * len(header))

        # Rows
        for s in self.samples:
            row = f"{s.name[:name_w]:<{name_w}} │ {s.log_likelihood:>9.2f} │ {s.perplexity:>7.1f} │ {s.rank_1_fraction:>5.0%} │ {s.avg_rank:>7.1f} │ {s.bleu_1:>6.2f} │ {s.rouge_1:>7.2f} │ {s.rouge_l:>7.2f}"
            if show_tokens:
                row += f" │ {s.num_tokens:>4}"
            lines.append(row)

        # Summary row
        lines.append("─" * len(header))
        summary_row = f"{'MEAN':<{name_w}} │ {self.mean_log_likelihood:>9.2f} │ {self.mean_perplexity:>7.1f} │ {self.mean_rank_1_fraction:>5.0%} │ {self.mean_avg_rank:>7.1f} │ {self.mean_bleu_1:>6.2f} │ {self.mean_rouge_1:>7.2f} │ {self.mean_rouge_l:>7.2f}"
        if show_tokens:
            total_tokens = sum(s.num_tokens for s in self.samples)
            summary_row += f" │ {total_tokens:>4}"
        lines.append(summary_row)
        lines.append("─" * len(header))

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "model_name": self.model_name,
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "device": self.device,
            "mean_log_likelihood": self.mean_log_likelihood,
            "mean_perplexity": self.mean_perplexity,
            "mean_rank_1_fraction": self.mean_rank_1_fraction,
            "mean_avg_rank": self.mean_avg_rank,
            "mean_bleu_1": self.mean_bleu_1,
            "mean_rouge_1": self.mean_rouge_1,
            "mean_rouge_l": self.mean_rouge_l,
            "samples": [s.to_dict() for s in self.samples],
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Results saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResults":
        """Load results from JSON file."""
        with open(path) as f:
            d = json.load(f)
        samples = [SampleResult.from_dict(s) for s in d.pop("samples")]
        # Handle older results without new fields
        d.setdefault("mean_bleu_1", 0.0)
        d.setdefault("mean_rouge_1", 0.0)
        d.setdefault("mean_rouge_l", 0.0)
        return cls(samples=samples, **d)

    def compare(self, other: "ExperimentResults") -> str:
        """Compare two experiment results."""
        lines = []
        lines.append(f"Comparison: {self.experiment_name} vs {other.experiment_name}")
        lines.append(f"Models: {self.model_name} vs {other.model_name}")
        lines.append("")

        lines.append(f"{'Metric':<20} │ {'This':>12} │ {'Other':>12} │ {'Diff':>12}")
        lines.append("─" * 62)

        metrics = [
            ("Mean Log-Lik", self.mean_log_likelihood, other.mean_log_likelihood),
            ("Mean Perplexity", self.mean_perplexity, other.mean_perplexity),
            (
                "Mean Rank-1 %",
                self.mean_rank_1_fraction * 100,
                other.mean_rank_1_fraction * 100,
            ),
            ("Mean Avg Rank", self.mean_avg_rank, other.mean_avg_rank),
            ("Mean BLEU-1", self.mean_bleu_1, other.mean_bleu_1),
            ("Mean ROUGE-1", self.mean_rouge_1, other.mean_rouge_1),
            ("Mean ROUGE-L", self.mean_rouge_l, other.mean_rouge_l),
        ]

        for name, v1, v2 in metrics:
            diff = v1 - v2
            sign = "+" if diff > 0 else ""
            lines.append(
                f"{name:<20} │ {v1:>12.2f} │ {v2:>12.2f} │ {sign}{diff:>11.2f}"
            )

        return "\n".join(lines)
