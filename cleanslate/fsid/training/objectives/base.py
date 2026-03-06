"""Base interfaces for FSID unlearning objectives."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class Objective(ABC):
    """Abstract base class for unlearning objectives."""

    name: str = ""
    needs_ref_logps: bool = False
    ref_logps_are_sums: bool = False
    owns_retain_loss: bool = False

    def setup(self, model: Any, config: Any) -> None:
        """Called once after the train model is prepared."""

    def teardown(self) -> None:
        """Called after training finishes."""

    @abstractmethod
    def compute_loss(
        self,
        model: Any,
        forget_batch: dict[str, torch.Tensor],
        retain_batch: dict[str, torch.Tensor] | None,
        retain_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the objective loss for one micro-step."""


def standard_retain_loss(
    model: Any,
    retain_batch: dict[str, torch.Tensor] | None,
    device: str | torch.device,
) -> torch.Tensor:
    """Standard CE retain loss on the retain batch."""
    if retain_batch is None:
        return torch.tensor(0.0, device=device)

    outputs = model(
        input_ids=retain_batch["input_ids"],
        attention_mask=retain_batch["attention_mask"],
        labels=retain_batch["labels"],
    )
    return outputs.loss
