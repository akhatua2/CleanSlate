"""Forget-set construction rule registry for FSID curation."""

from dataclasses import dataclass
from typing import Any, Callable

from cleanslate.fsid.rules.baseline import (
    ARTIFACT_RULE_NAME as BASELINE_ARTIFACT_RULE_NAME,
    RULE_NAME as BASELINE_RULE_NAME,
    curate_baseline_sets,
    save_curated_sets,
)

CurateRuleResult = tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]
CurateRuleFn = Callable[..., CurateRuleResult]


@dataclass(frozen=True)
class ForgetSetRule:
    """Named forget-set construction rule."""

    name: str
    artifact_rule_name: str
    curate: CurateRuleFn


FORGET_SET_RULES: dict[str, ForgetSetRule] = {
    BASELINE_RULE_NAME: ForgetSetRule(
        name=BASELINE_RULE_NAME,
        artifact_rule_name=BASELINE_ARTIFACT_RULE_NAME,
        curate=curate_baseline_sets,
    )
}


def list_forget_set_rules() -> tuple[str, ...]:
    """Return the supported forget-set rule names."""
    return tuple(sorted(FORGET_SET_RULES))


def get_forget_set_rule(name: str) -> ForgetSetRule:
    """Resolve a forget-set construction rule by CLI name."""
    normalized = name.strip().lower()
    rule = FORGET_SET_RULES.get(normalized)
    if rule is None:
        allowed = ", ".join(list_forget_set_rules())
        raise ValueError(
            f"Unknown forget-set rule {name!r}. Allowed rules: {allowed}"
        )
    return rule
