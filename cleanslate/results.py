"""Structured results for experiments."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


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

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SampleResult":
        return cls(**d)


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

    def __post_init__(self):
        """Compute aggregate statistics."""
        if self.samples:
            self.mean_log_likelihood = sum(
                s.log_likelihood for s in self.samples
            ) / len(self.samples)
            self.mean_perplexity = sum(s.perplexity for s in self.samples) / len(
                self.samples
            )
            self.mean_rank_1_fraction = sum(
                s.rank_1_fraction for s in self.samples
            ) / len(self.samples)
            self.mean_avg_rank = sum(s.avg_rank for s in self.samples) / len(
                self.samples
            )

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
        header = f"{'Sample':<{name_w}} │ {'Log-Lik':>10} │ {'PPL':>8} │ {'Rank-1':>8} │ {'AvgRank':>8}"
        if show_tokens:
            header += f" │ {'Tokens':>6}"
        lines.append("─" * len(header))
        lines.append(header)
        lines.append("─" * len(header))

        # Rows
        for s in self.samples:
            row = f"{s.name[:name_w]:<{name_w}} │ {s.log_likelihood:>10.2f} │ {s.perplexity:>8.2f} │ {s.rank_1_fraction:>7.1%} │ {s.avg_rank:>8.1f}"
            if show_tokens:
                row += f" │ {s.num_tokens:>6}"
            lines.append(row)

        # Summary row
        lines.append("─" * len(header))
        summary_row = f"{'MEAN':<{name_w}} │ {self.mean_log_likelihood:>10.2f} │ {self.mean_perplexity:>8.2f} │ {self.mean_rank_1_fraction:>7.1%} │ {self.mean_avg_rank:>8.1f}"
        if show_tokens:
            total_tokens = sum(s.num_tokens for s in self.samples)
            summary_row += f" │ {total_tokens:>6}"
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
            "samples": [s.to_dict() for s in self.samples],
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentResults":
        """Load results from JSON file."""
        with open(path) as f:
            d = json.load(f)
        samples = [SampleResult.from_dict(s) for s in d.pop("samples")]
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
        ]

        for name, v1, v2 in metrics:
            diff = v1 - v2
            sign = "+" if diff > 0 else ""
            lines.append(
                f"{name:<20} │ {v1:>12.2f} │ {v2:>12.2f} │ {sign}{diff:>11.2f}"
            )

        return "\n".join(lines)
