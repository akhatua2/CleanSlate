"""Simplified NPO objective."""

from typing import Any

import torch
import torch.nn.functional as F

from cleanslate.fsid.training.objectives.base import Objective, standard_retain_loss
from cleanslate.fsid.training.objectives.utils import conditional_avg_log_probs


class SimNPOObjective(Objective):
    """Reference-free NPO variant."""

    name = "simnpo"

    def __init__(self, beta: float = 0.1, gamma: float = 0.1) -> None:
        self.beta = beta
        self.gamma = gamma

    def compute_loss(
        self,
        model: Any,
        forget_batch: dict[str, torch.Tensor],
        retain_batch: dict[str, torch.Tensor] | None,
        retain_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = forget_batch["input_ids"].device
        avg_log_prob, _ = conditional_avg_log_probs(
            model,
            forget_batch["input_ids"],
            forget_batch["attention_mask"],
            forget_batch["seed_lens"],
        )
        forget_loss = -(2.0 / self.beta) * F.logsigmoid(
            -self.beta * avg_log_prob - self.gamma
        ).mean()
        retain_loss = standard_retain_loss(model, retain_batch, device)
        total_loss = forget_loss + retain_weight * retain_loss
        return total_loss, forget_loss.detach(), retain_loss.detach()
