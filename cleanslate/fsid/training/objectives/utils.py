"""Objective math utilities shared across unlearning methods."""

from typing import Any

import torch
import torch.nn.functional as F


def build_logprob_target_mask(
    seq_len: int,
    seed_lens: torch.Tensor,
    full_lens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Mask shifted-logit positions whose predictions belong to the forget suffix."""
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    start = (seed_lens - 1).clamp(min=0).unsqueeze(1)
    end = (full_lens - 1).clamp(min=0).unsqueeze(1)
    return (positions >= start) & (positions < end)


def conditional_avg_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seed_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-averaged log P(target | seed) for a padded batch."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    next_tokens = input_ids[:, 1:]

    token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
        -1, next_tokens.unsqueeze(-1)
    ).squeeze(-1)

    full_lens = attention_mask.sum(dim=1)
    mask = build_logprob_target_mask(
        token_log_probs.size(1),
        seed_lens,
        full_lens,
        input_ids.device,
    )

    num_tokens = mask.sum(dim=1).clamp(min=1)
    sum_log_prob = (token_log_probs * mask.to(token_log_probs.dtype)).sum(dim=1)
    return sum_log_prob / num_tokens.float(), num_tokens


def conditional_sum_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seed_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-sequence SUM log P(target | seed) for a padded batch."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    next_tokens = input_ids[:, 1:]

    token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
        -1, next_tokens.unsqueeze(-1)
    ).squeeze(-1)

    full_lens = attention_mask.sum(dim=1)
    mask = build_logprob_target_mask(
        token_log_probs.size(1),
        seed_lens,
        full_lens,
        input_ids.device,
    )

    num_tokens = mask.sum(dim=1).clamp(min=1)
    sum_log_prob = (token_log_probs * mask.to(token_log_probs.dtype)).sum(dim=1)
    return sum_log_prob, num_tokens
