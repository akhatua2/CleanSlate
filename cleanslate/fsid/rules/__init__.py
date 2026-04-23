"""Forget-set construction rule registry for FSID curation.

Every rule emits forget rows with a fixed (prefix, suffix, content_id, request_id,
rule_name, ...) schema, so the trainer tokenizes them uniformly via
`cleanslate.fsid.datasets.unlearning.build_forget_pairs`. Rules differ only in how
the windows are chosen (character-sliding vs. retrieval-localized vs. pre-sweep).

Registry keys are `<family>-<corpus>` so selecting a rule pins both the
mechanism and the HF retrieval dataset.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from cleanslate.fsid.rules import baseline, bm25, corpus_sweep, igm
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


def _wrap(
    fn: CurateRuleFn,
    *,
    dataset_id: str | None,
    artifact_rule_name: str,
) -> CurateRuleFn:
    """Pin a default `base_dataset` and stamp `rule_name` on all emitted rows."""

    def wrapped(
        unlearn_song_records: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CurateRuleResult:
        if dataset_id is not None and not kwargs.get("base_dataset"):
            kwargs["base_dataset"] = dataset_id
        forget, retain = fn(unlearn_song_records, **kwargs)
        for row in forget:
            row["rule_name"] = artifact_rule_name
        for row in retain:
            row["rule_name"] = artifact_rule_name
        return forget, retain

    return wrapped


_RULE_SPECS: tuple[tuple[str, CurateRuleFn, str | None, str], ...] = (
    ("baseline", baseline.curate_baseline_sets, None, "baseline_literal"),
    (
        "bm25-10B",
        bm25.curate_bm25_sets,
        "unlearning-cleanslate/verif-bm25-dolma-10B",
        "bm25_dolma_10B",
    ),
    (
        "bm25-6T",
        bm25.curate_bm25_sets,
        "unlearning-cleanslate/verif-bm25-dolma-6T",
        "bm25_dolma_6T",
    ),
    (
        "igm-10B",
        igm.curate_igm_sets,
        "unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B",
        "igm_dolma_10B",
    ),
    (
        "igm-6T-eval-sweep",
        corpus_sweep.curate_corpus_sweep_sets,
        "unlearning-cleanslate/corpus_sweep_cleanslate_dataset",
        "corpus_sweep_dolma",
    ),
    (
        "igm-dclm",
        igm.curate_igm_sets,
        "unlearning-cleanslate/infini_gram_mini_cleanslate_dclm",
        "igm_dclm",
    ),
)


FORGET_SET_RULES: dict[str, ForgetSetRule] = {
    name: ForgetSetRule(
        name=name,
        artifact_rule_name=artifact,
        curate=_wrap(fn, dataset_id=ds, artifact_rule_name=artifact),
    )
    for name, fn, ds, artifact in _RULE_SPECS
}


def list_forget_set_rules() -> tuple[str, ...]:
    """Return the supported forget-set rule names, registration-ordered."""
    return tuple(FORGET_SET_RULES)


def get_forget_set_rule(name: str) -> ForgetSetRule:
    """Resolve a rule by CLI name."""
    rule = FORGET_SET_RULES.get(name.strip())
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
