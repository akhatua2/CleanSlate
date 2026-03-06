"""Objective registry for FSID local unlearning."""

from typing import Any

from cleanslate.fsid.training.objectives.base import Objective
from cleanslate.fsid.training.objectives.ga import GAObjective
from cleanslate.fsid.training.objectives.npo import NPOObjective
from cleanslate.fsid.training.objectives.simnpo import SimNPOObjective


OBJECTIVE_REGISTRY: dict[str, type[Objective]] = {
    "ga": GAObjective,
    "grad_diff": GAObjective,
    "npo": NPOObjective,
    "simnpo": SimNPOObjective,
}


def list_objectives() -> list[str]:
    """Return supported objective names."""
    return sorted(OBJECTIVE_REGISTRY)


def build_objective(config: Any) -> Objective:
    """Instantiate an objective from a TrainerConfig."""
    name = config.objective
    if name not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective '{name}'. Available: {sorted(OBJECTIVE_REGISTRY)}"
        )

    cls = OBJECTIVE_REGISTRY[name]
    if name == "npo":
        return cls(beta=config.npo_beta)
    if name == "simnpo":
        return cls(beta=config.npo_beta, gamma=config.simnpo_gamma)
    return cls()
