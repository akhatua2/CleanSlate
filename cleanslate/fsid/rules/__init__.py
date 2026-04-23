"""Forget-set construction rule registry for FSID curation.

Every rule emits forget rows with a fixed (prefix, suffix, content_id, request_id,
rule_name, ...) schema, so the trainer tokenizes them uniformly via
`cleanslate.fsid.datasets.unlearning.build_forget_pairs`. Rules differ only in how
the windows are chosen (character-sliding vs. corpus-matched).
"""

from dataclasses import dataclass
from typing import Any, Protocol

from cleanslate.fsid.rules import baseline, corpus_sweep
from cleanslate.fsid.rules.baseline import save_curated_sets

type CurateRuleResult = tuple[list[dict[str, Any]], list[dict[str, Any]]]


class CurateRuleFn(Protocol):
    def __call__(
        self,
        unlearn_song_records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurateRuleResult: ...


@dataclass(frozen=True)
class ForgetSetRule:
    """Named rule: produces (forget_rows, retain_rows) at curation time."""

    name: str
    artifact_rule_name: str
    curate: CurateRuleFn


FORGET_SET_RULES: dict[str, ForgetSetRule] = {
    baseline.RULE_NAME: ForgetSetRule(
        name=baseline.RULE_NAME,
        artifact_rule_name=baseline.ARTIFACT_RULE_NAME,
        curate=baseline.curate_baseline_sets,
    ),
    corpus_sweep.RULE_NAME: ForgetSetRule(
        name=corpus_sweep.RULE_NAME,
        artifact_rule_name=corpus_sweep.ARTIFACT_RULE_NAME,
        curate=corpus_sweep.curate_corpus_sweep_sets,
    ),
}


def list_forget_set_rules() -> tuple[str, ...]:
    """Return the supported forget-set rule names, sorted."""
    return tuple(sorted(FORGET_SET_RULES))


def get_forget_set_rule(name: str) -> ForgetSetRule:
    """Resolve a rule by CLI name."""
    rule = FORGET_SET_RULES.get(name.strip().lower())
    if rule is None:
        raise ValueError(
            f"Unknown forget-set rule {name!r}. Allowed: {', '.join(list_forget_set_rules())}"
        )
    return rule


__all__ = [
    "ForgetSetRule",
    "FORGET_SET_RULES",
    "list_forget_set_rules",
    "get_forget_set_rule",
    "save_curated_sets",
]
