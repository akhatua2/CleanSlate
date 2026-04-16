"""Loss utilities for FSID unlearning, ported from open-unlearning patterns."""

from typing import Any

import torch
import torch.nn.functional as F


def compute_batch_nll(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Per-sample NLL (sum of CE over non-masked tokens).

    Returns a (batch_size,) tensor of per-sample summed negative log-likelihoods.
    Follows open-unlearning's compute_batch_nll pattern: compute per-token CE,
    mask out positions where labels == -100, and sum per sample.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    target = labels[:, 1:]

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(target.shape)

    mask = target != -100
    per_sample_nll = (per_token_loss * mask).sum(dim=1)
    return per_sample_nll


def compute_npo_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """NPO loss: -2/beta * logsigmoid(-beta * (log_pi - log_ref)).

    Uses summed log-probs (not averaged) per the original NPO formulation.
    """
    # Current model log-probs (negative NLL = log-prob)
    policy_nll = compute_batch_nll(model, input_ids, attention_mask, labels)
    policy_log_probs = -policy_nll

    # Reference model log-probs
    with torch.no_grad():
        ref_nll = compute_batch_nll(ref_model, input_ids, attention_mask, labels)
    ref_log_probs = -ref_nll

    loss = -(2.0 / beta) * F.logsigmoid(
        -beta * (policy_log_probs - ref_log_probs)
    ).mean()
    return loss


def compute_simnpo_loss(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """SimNPO loss: reference-free NPO using per-sample avg log-probs.

    Matches open-unlearning convention: compute summed NLL via
    compute_batch_nll, then explicitly divide by token count here
    so the averaging is visible and auditable.
    """
    per_sample_sum_nll = compute_batch_nll(model, input_ids, attention_mask, labels)
    token_counts = (labels[:, 1:] != -100).sum(dim=1).clamp(min=1)
    avg_log_probs = -per_sample_sum_nll / token_counts.float()

    loss = -(2.0 / beta) * F.logsigmoid(
        beta * (avg_log_probs - gamma)
    ).mean()
    return loss


def compute_kl_retain_loss(
    model: Any,
    ref_model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """KL divergence retain loss: KL(ref || model) on non-masked tokens.

    Matches open-unlearning's retain_loss_type=KL: trains the model to
    preserve the reference model's output distribution on retain data.
    """
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
    ref_logits = ref_outputs.logits[:, :-1, :].float()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    model_logits = outputs.logits[:, :-1, :].float()

    target = labels[:, 1:]
    mask = (target != -100).float()

    # KL(ref || model) per token, then mask and average
    ref_probs = F.softmax(ref_logits, dim=-1)
    model_log_probs = F.log_softmax(model_logits, dim=-1)

    # per-token KL: sum over vocab dimension
    per_token_kl = F.kl_div(
        model_log_probs, ref_probs, reduction="none", log_target=False
    ).sum(dim=-1)

    # mask out padding/ignored positions and average
    num_tokens = mask.sum().clamp(min=1)
    loss = (per_token_kl * mask).sum() / num_tokens
    return loss
