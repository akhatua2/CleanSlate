"""ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric implementation.

ROUGE measures overlap between predicted and target text, focusing on recall.

Plain English:
    While BLEU asks "how much of prediction is in target?", ROUGE also asks
    "how much of target is covered by prediction?" and combines both into
    an F1 score (harmonic mean of precision and recall).

    ROUGE-1: Word-level overlap
        How many individual words are shared between prediction and target?

    ROUGE-2: Bigram overlap
        How many consecutive word pairs are shared?
        More strict than ROUGE-1 because word order matters.

    ROUGE-L: Longest Common Subsequence
        What's the longest sequence of words that appears in both texts
        (in the same order, but not necessarily consecutive)?

        "the quick brown fox" vs "the brown fox jumps"
        LCS = "the brown fox" (length 3)

        This captures fluency and structure, not just vocabulary.

    Higher ROUGE = more overlap = prediction is closer to target.
"""

from collections import Counter

from .base import Metric, MetricResult


def lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


class ROUGE1(Metric):
    """ROUGE-1: Unigram (single word) overlap.

    Checks: What fraction of words are shared between prediction and target?
    Returns: F1 score (balances precision and recall).

    Good for: Basic vocabulary overlap. If prediction and target use the
    same words, score is high, regardless of order.
    """

    @property
    def name(self) -> str:
        return "ROUGE-1"

    def compute(
        self,
        predicted: list[str],
        target: list[str],
    ) -> MetricResult:
        """Compute ROUGE-1 F1 score."""
        if not predicted or not target:
            return MetricResult(name=self.name, score=0.0)

        pred_counts = Counter(predicted)
        target_counts = Counter(target)

        # Count overlapping unigrams
        overlap = 0
        for token, count in pred_counts.items():
            overlap += min(count, target_counts.get(token, 0))

        # Precision and recall
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(target) if target else 0.0

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return MetricResult(name=self.name, score=f1)


class ROUGE2(Metric):
    """ROUGE-2: Bigram (consecutive word pair) overlap.

    Checks: What fraction of word pairs are shared?
    Returns: F1 score.

    Good for: Detecting phrase-level similarity. Stricter than ROUGE-1
    because adjacent words must match together.
    """

    @property
    def name(self) -> str:
        return "ROUGE-2"

    def compute(
        self,
        predicted: list[str],
        target: list[str],
    ) -> MetricResult:
        """Compute ROUGE-2 F1 score."""
        if len(predicted) < 2 or len(target) < 2:
            return MetricResult(name=self.name, score=0.0)

        # Get bigrams
        pred_bigrams = Counter(
            (predicted[i], predicted[i + 1]) for i in range(len(predicted) - 1)
        )
        target_bigrams = Counter(
            (target[i], target[i + 1]) for i in range(len(target) - 1)
        )

        # Count overlapping bigrams
        overlap = 0
        for bigram, count in pred_bigrams.items():
            overlap += min(count, target_bigrams.get(bigram, 0))

        # Precision and recall
        precision = overlap / sum(pred_bigrams.values()) if pred_bigrams else 0.0
        recall = overlap / sum(target_bigrams.values()) if target_bigrams else 0.0

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return MetricResult(name=self.name, score=f1)


class ROUGEL(Metric):
    """ROUGE-L: Longest Common Subsequence (LCS) based metric.

    Checks: What's the longest sequence of words appearing in both texts
    in the same order (not necessarily consecutive)?

    Example:
        pred:   "the quick brown fox"
        target: "the brown fox jumps"
        LCS:    "the brown fox" (length 3)

    Good for: Measuring structural/fluency similarity. Captures when
    the overall flow matches even if some words differ.
    """

    @property
    def name(self) -> str:
        return "ROUGE-L"

    def compute(
        self,
        predicted: list[str],
        target: list[str],
    ) -> MetricResult:
        """Compute ROUGE-L F1 score."""
        if not predicted or not target:
            return MetricResult(name=self.name, score=0.0)

        lcs_len = lcs_length(predicted, target)

        # Precision and recall based on LCS
        precision = lcs_len / len(predicted) if predicted else 0.0
        recall = lcs_len / len(target) if target else 0.0

        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return MetricResult(name=self.name, score=f1)
