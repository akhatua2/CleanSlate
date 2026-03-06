"""Negative Preference Optimization objective."""

from typing import Any

import torch
import torch.nn.functional as F

from cleanslate.fsid.training.objectives.base import Objective, standard_retain_loss
from cleanslate.fsid.training.objectives.utils import conditional_sum_log_probs


class NPOObjective(Objective):
    """Negative Preference Optimization."""

    name = "npo"
    needs_ref_logps = True
    ref_logps_are_sums = True

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute_loss(
        self,
        model: Any,
        forget_batch: dict[str, torch.Tensor],
        retain_batch: dict[str, torch.Tensor] | None,
        retain_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = forget_batch["input_ids"].device
        sum_log_prob, _ = conditional_sum_log_probs(
            model,
            forget_batch["input_ids"],
            forget_batch["attention_mask"],
            forget_batch["seed_lens"],
        )
        ref_sum_log_prob = forget_batch["ref_logps"]

        forget_loss = -(2.0 / self.beta) * F.logsigmoid(
            -self.beta * (sum_log_prob - ref_sum_log_prob)
        ).mean()
        retain_loss = standard_retain_loss(model, retain_batch, device)
        total_loss = forget_loss + retain_weight * retain_loss
        return total_loss, forget_loss.detach(), retain_loss.detach()
