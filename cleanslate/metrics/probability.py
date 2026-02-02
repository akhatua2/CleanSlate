"""Probability-based metrics for memorization/extraction analysis.

Based on Hayes et al. (n, p)-discoverable extraction framework.

Plain English:
    These metrics measure how likely a model is to generate specific text.
    
    pz (extraction probability):
        "If I give the model this prompt, what's the probability it outputs
        exactly this text?" Computed by multiplying the probability of each
        token given the previous tokens.
        
        Example: P("brown fox" | "the quick") = P("brown"|"the quick") * P("fox"|"the quick brown")
        
        High pz = model is very likely to produce this exact text = memorized.
        
    Perplexity:
        "How surprised is the model by this text?" 
        Lower = less surprised = more memorized.
        It's the geometric mean of 1/probability across all tokens.
        
        Perplexity of 1 = model is 100% confident about every token.
        Perplexity of 100 = on average, model thinks 100 tokens are equally likely.
        
    (n, p)-discoverable extraction:
        "If I prompt the model n times, what's my chance of getting this exact text?"
        
        Like coin flipping: if each flip has 10% chance of heads (pz=0.1),
        flipping 10 times gives you 65% chance of at least one heads.
        
        Formula: P(at least one success in n tries) = 1 - (1 - pz)^n
        
        This helps answer: "How many attempts would an attacker need to
        extract this memorized content with 90% confidence?"
"""

import math
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Result from extraction probability computation.
    
    Attributes:
        log_prob: Sum of log probabilities (more numerically stable than prob).
                  More negative = less likely.
        prob: The extraction probability pz = exp(log_prob).
              Range [0, 1]. Higher = model more likely to generate this exact text.
        perplexity: How "surprised" the model is. Lower = less surprised = memorized.
                    Perplexity of 10 means model thinks ~10 tokens equally likely on average.
        num_tokens: Number of tokens in the target sequence.
    """
    
    log_prob: float
    prob: float
    perplexity: float
    num_tokens: int
    
    def prompts_needed(self, p: float = 0.9) -> float | None:
        """
        Compute n: number of prompts needed to extract with probability >= p.
        
        Uses: n >= log(1-p) / log(1-pz)
        
        Args:
            p: Target probability of successful extraction (default 0.9 = 90%)
            
        Returns:
            Number of prompts needed, or None if pz is 0 or 1
        """
        if self.prob <= 0 or self.prob >= 1:
            return None
        return math.ceil(math.log(1 - p) / math.log(1 - self.prob))
    
    def extraction_prob_after_n(self, n: int) -> float:
        """
        Compute probability of extraction after n independent prompts.
        
        Uses: p = 1 - (1 - pz)^n
        
        Args:
            n: Number of independent prompts
            
        Returns:
            Probability of extracting at least once in n attempts
        """
        if self.prob <= 0:
            return 0.0
        if self.prob >= 1:
            return 1.0
        return 1 - math.pow(1 - self.prob, n)


def compute_extraction_probability(
    token_log_probs: list[float],
) -> ExtractionResult:
    """
    Compute extraction probability from per-token log probabilities.
    
    This is the pz metric from Hayes et al. - the probability of generating
    a specific suffix given a prefix.
    
    Args:
        token_log_probs: List of conditional log probabilities for each token
        
    Returns:
        ExtractionResult with pz and derived metrics
    """
    num_tokens = len(token_log_probs)
    if num_tokens == 0:
        return ExtractionResult(
            log_prob=0.0,
            prob=1.0,
            perplexity=1.0,
            num_tokens=0,
        )
    
    # Check for impossible tokens (from top-k filtering)
    if any(not math.isfinite(lp) for lp in token_log_probs):
        return ExtractionResult(
            log_prob=float("-inf"),
            prob=0.0,
            perplexity=float("inf"),
            num_tokens=num_tokens,
        )
    
    log_prob = sum(token_log_probs)
    prob = math.exp(log_prob)
    avg_neg_log_prob = -log_prob / num_tokens
    perplexity = math.exp(avg_neg_log_prob)
    
    return ExtractionResult(
        log_prob=log_prob,
        prob=prob,
        perplexity=perplexity,
        num_tokens=num_tokens,
    )


def prompts_for_probability(pz: float, p: float = 0.9) -> float | None:
    """
    Compute number of prompts n needed to extract with probability >= p.
    
    From Hayes et al.: n >= log(1-p) / log(1-pz)
    
    Args:
        pz: Single-prompt extraction probability
        p: Target cumulative probability (default 0.9)
        
    Returns:
        Minimum n, or None if pz is 0 or 1
    """
    if pz <= 0 or pz >= 1:
        return None
    return math.ceil(math.log(1 - p) / math.log(1 - pz))


def probability_after_n_prompts(pz: float, n: int) -> float:
    """
    Compute probability of extraction after n independent prompts.
    
    From Hayes et al.: p = 1 - (1 - pz)^n
    
    Like coin flipping: if each flip has probability pz of success,
    after n flips, what's the probability of at least one success?
    
    Args:
        pz: Single-prompt extraction probability  
        n: Number of independent prompts
        
    Returns:
        Probability of at least one successful extraction
    """
    if pz <= 0:
        return 0.0
    if pz >= 1:
        return 1.0
    return 1 - math.pow(1 - pz, n)
