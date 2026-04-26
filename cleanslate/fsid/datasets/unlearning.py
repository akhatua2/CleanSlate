"""Tokenized forget/retain datasets and collators for unlearning."""

from typing import Any

from torch.utils.data import Dataset as TorchDataset


class ForgetPairDataset(TorchDataset[dict[str, Any]]):
    """Tokenized (prefix, suffix) forget pairs with seed masking baked into labels.

    Labels use -100 for prefix (seed) positions so standard CE loss ignores them.
    """

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._pairs[index]
        input_ids: list[int] = item["input_ids"]
        seed_len: int = item["seed_len"]
        labels = [-100] * seed_len + input_ids[seed_len:]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class TextSequenceDataset(TorchDataset[dict[str, Any]]):
    """Tokenized text sequences for retain/validation next-token modeling (labels == input_ids)."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        input_ids: list[int] = self._examples[index]["input_ids"]
        return {
            "input_ids": input_ids,
            "labels": list(input_ids),
            "attention_mask": [1] * len(input_ids),
        }


def build_forget_pairs(
    forget_rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Tokenize pre-sliced (prefix, suffix) forget rows.

    Both `baseline` and `corpus_sweep` emit forget rows with explicit `prefix` and
    `suffix` columns; this function just tokenizes them — no windowing at train time.
    """
    pairs: list[dict[str, Any]] = []
    for row in forget_rows:
        prefix_ids = tokenizer.encode(row["prefix"], add_special_tokens=False)
        suffix_ids = tokenizer.encode(row["suffix"], add_special_tokens=False)
        input_ids = prefix_ids + suffix_ids
        seed_len = min(len(prefix_ids), len(input_ids) - 1)
        if seed_len <= 0:
            raise ValueError(
                f"window content_id={row.get('content_id')!r} window_idx={row.get('window_idx')!r} "
                f"yields empty target (prefix_tokens={len(prefix_ids)}, suffix_tokens={len(suffix_ids)})"
            )
        pairs.append(
            {
                "input_ids": input_ids,
                "seed_len": seed_len,
                "content_id": row["content_id"],
                "request_id": row["request_id"],
            }
        )
    return pairs


def tokenize_text_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """Tokenize raw text rows from a retain/validation artifact ({"text": ..., ...})."""
    out: list[dict[str, Any]] = []
    for row in rows:
        text: str = row["text"]
        if not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) >= 2:
            out.append({"input_ids": token_ids})
    return out


