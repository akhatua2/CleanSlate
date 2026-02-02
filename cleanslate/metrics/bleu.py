"""BLEU (Bilingual Evaluation Understudy) metric implementation.

BLEU measures how many words/phrases from the prediction appear in the target.

Plain English:
    Imagine you're comparing two sentences. BLEU asks: "How many of the
    predicted words (or pairs of words, or triplets...) also appear in 
    the target?" It's like checking how much vocabulary overlap there is.
    
    BLEU-1: Checks individual words (unigrams)
        "the cat sat" vs "the dog sat" → 2/3 words match = 0.67
    
    BLEU-2: Also checks pairs of consecutive words (bigrams)
        "the cat sat" has bigrams: ("the","cat"), ("cat","sat")
        "the dog sat" has bigrams: ("the","dog"), ("dog","sat")
        No bigram matches → lower score
        
    Higher BLEU = more overlap = prediction is closer to target.
    
    The "modified precision" clips counts so repeating a word doesn't
    artificially inflate the score.
"""

from collections import Counter
import math

from .base import Metric, MetricResult


def get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-grams from a list of tokens."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def modified_precision(predicted: list[str], target: list[str], n: int) -> float:
    """
    Compute modified n-gram precision.
    
    Clips counts to reference counts to avoid gaming by repetition.
    """
    pred_ngrams = get_ngrams(predicted, n)
    target_ngrams = get_ngrams(target, n)
    
    if not pred_ngrams:
        return 0.0
    
    # Clip predicted counts to reference counts
    clipped_count = 0
    total_count = 0
    
    for ngram, count in pred_ngrams.items():
        clipped_count += min(count, target_ngrams.get(ngram, 0))
        total_count += count
    
    return clipped_count / total_count if total_count > 0 else 0.0


def brevity_penalty(predicted: list[str], target: list[str]) -> float:
    """Compute brevity penalty for short predictions."""
    pred_len = len(predicted)
    target_len = len(target)
    
    if pred_len >= target_len:
        return 1.0
    elif pred_len == 0:
        return 0.0
    else:
        return math.exp(1 - target_len / pred_len)


class BLEU(Metric):
    """
    BLEU score implementation.
    
    BLEU measures n-gram precision between predicted and target text.
    Higher = more overlap = model predicts closer to target.
    
    Args:
        n: Maximum n-gram order (default: 4 for BLEU-4)
        weights: Weights for each n-gram order (default: uniform)
    """
    
    def __init__(self, n: int = 4, weights: list[float] | None = None):
        self.n = n
        self.weights = weights or [1.0 / n] * n
    
    @property
    def name(self) -> str:
        return f"BLEU-{self.n}"
    
    def compute(
        self,
        predicted: list[str],
        target: list[str],
    ) -> MetricResult:
        """Compute BLEU score."""
        if not predicted or not target:
            return MetricResult(name=self.name, score=0.0)
        
        # Compute modified precision for each n-gram order
        precisions = []
        for i in range(1, self.n + 1):
            p = modified_precision(predicted, target, i)
            precisions.append(p)
        
        # If any precision is 0, BLEU is 0
        if any(p == 0 for p in precisions):
            return MetricResult(name=self.name, score=0.0)
        
        # Geometric mean of precisions
        log_precisions = [w * math.log(p) for w, p in zip(self.weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
        
        # Apply brevity penalty
        bp = brevity_penalty(predicted, target)
        score = bp * geo_mean
        
        return MetricResult(name=self.name, score=score)


class BLEU1(BLEU):
    """BLEU-1: Unigram (single word) precision only.
    
    Checks: What fraction of predicted words appear in the target?
    Good for: Basic vocabulary overlap, ignores word order completely.
    """
    
    def __init__(self):
        super().__init__(n=1, weights=[1.0])
    
    @property
    def name(self) -> str:
        return "BLEU-1"


class BLEU2(BLEU):
    """BLEU-2: Unigram + bigram (word pair) precision.
    
    Checks: Word overlap AND consecutive word pair overlap.
    Good for: Catching when word order matters (higher bar than BLEU-1).
    """
    
    def __init__(self):
        super().__init__(n=2, weights=[0.5, 0.5])
    
    @property
    def name(self) -> str:
        return "BLEU-2"
