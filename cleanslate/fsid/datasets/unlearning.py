"""Training-data preparation and collators for FSID unlearning."""

from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


class ForgetPairDataset(TorchDataset):
    """Tokenized forget pairs with labels-based seed masking.

    Returns input_ids, labels (with -100 for seed tokens), and attention_mask.
    The masking is baked into labels so standard CE loss automatically
    ignores seed tokens — no need for seed_lens at training time.
    """

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._pairs[index]
        input_ids = item["input_ids"]
        seed_len = item["seed_len"]
        # Mask seed tokens with -100 so CE loss ignores them
        labels = [-100] * seed_len + input_ids[seed_len:]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class TextSequenceDataset(TorchDataset):
    """Tokenized text sequences for retain/validation language modeling.

    All tokens are training targets (labels = input_ids).
    """

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self._examples[index]
        input_ids = example["input_ids"]
        return {
            "input_ids": input_ids,
            "labels": list(input_ids),
            "attention_mask": [1] * len(input_ids),
        }


def build_forget_pairs(
    forget_rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    text_col: str,
    window_size: int,
    stride: int,
    max_windows_per_song: int | None,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Create local forget training pairs from lyrics via sliding windows."""
    window_total = window_size * 2
    pairs: list[dict[str, Any]] = []

    for row in forget_rows:
        text = row.get(text_col)
        if text is None:
            raise KeyError(
                f"Forget row missing required text column '{text_col}'. "
                f"Available keys: {sorted(row.keys())}. "
                "Pass the correct column name via forget_text_col."
            )
        if not isinstance(text, str) or not text.strip():
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < window_total:
            continue

        starts = list(range(0, len(token_ids) - window_total + 1, stride))
        if max_windows_per_song is not None and len(starts) > max_windows_per_song:
            step = len(starts) / max_windows_per_song
            starts = [
                starts[min(len(starts) - 1, int(index * step))]
                for index in range(max_windows_per_song)
            ]

        for start in starts:
            prefix = token_ids[start : start + window_size]
            suffix = token_ids[start + window_size : start + window_total]
            input_ids = (prefix + suffix)[:max_seq_len]
            seed_len = min(len(prefix), len(input_ids) - 1)
            if seed_len <= 0 or len(input_ids) <= seed_len:
                continue

            pairs.append(
                {
                    "input_ids": input_ids,
                    "seed_len": seed_len,
                    "content_id": row.get("content_id", ""),
                    "request_id": row.get("request_id", ""),
                }
            )

    return pairs


def tokenize_text_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    text_col: str,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    """Tokenize local texts for retain/validation next-token modeling."""
    examples: list[dict[str, Any]] = []
    for row in rows:
        text = row.get(text_col)
        if text is None:
            raise KeyError(
                f"Row missing required text column '{text_col}'. "
                f"Available keys: {sorted(row.keys())}. "
                "Pass the correct column name via CLI or YAML."
            )
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_seq_len]
        if len(token_ids) >= 2:
            examples.append({"input_ids": token_ids})
    return examples


def collate_forget(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    """Collate function for forget pairs with labels-based masking."""
    max_len = max(len(row["input_ids"]) for row in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for index, row in enumerate(batch):
        ids = row["input_ids"]
        seq_len = len(ids)
        input_ids[index, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :seq_len] = 1
        labels[index, :seq_len] = torch.tensor(row["labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_text_sequences(
    batch: list[dict[str, Any]],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Collate function for retain/validation language-modeling examples."""
    max_len = max(len(row["input_ids"]) for row in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for index, row in enumerate(batch):
        ids = row["input_ids"]
        seq_len = len(ids)
        input_ids[index, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :seq_len] = 1
        row_labels = row.get("labels", ids)
        labels[index, :seq_len] = torch.tensor(row_labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
