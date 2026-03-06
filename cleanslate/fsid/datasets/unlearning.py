"""Training-data preparation and collators for FSID unlearning."""

from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


class ForgetPairDataset(TorchDataset):
    """Tokenized forget pairs with optional cached reference log-probabilities."""

    def __init__(self, pairs: list[dict[str, Any]]) -> None:
        self._pairs = pairs
        self._ref_logps = [0.0] * len(pairs)

    def set_reference_logps(self, values: list[float]) -> None:
        if len(values) != len(self._pairs):
            raise ValueError("Reference log-prob list length mismatch.")
        self._ref_logps = values

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._pairs[index]
        return {
            "index": index,
            "input_ids": item["input_ids"],
            "seed_len": item["seed_len"],
            "song_id": item["song_id"],
            "request_id": item["request_id"],
            "ref_logp": self._ref_logps[index],
        }


class TextSequenceDataset(TorchDataset):
    """Tokenized text sequences for retain/validation language modeling."""

    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self._examples[index]
        return {
            "index": index,
            "input_ids": example["input_ids"],
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
                    "song_id": row.get("song_id", ""),
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
    """Collate function for forget pairs."""
    max_len = max(len(row["input_ids"]) for row in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    seed_lens = torch.zeros(batch_size, dtype=torch.long)
    indices = torch.zeros(batch_size, dtype=torch.long)
    ref_logps = torch.zeros(batch_size, dtype=torch.float32)

    for index, row in enumerate(batch):
        seq = torch.tensor(row["input_ids"], dtype=torch.long)
        seq_len = seq.numel()
        input_ids[index, :seq_len] = seq
        attention_mask[index, :seq_len] = 1
        seed_lens[index] = int(row["seed_len"])
        indices[index] = int(row["index"])
        ref_logps[index] = float(row["ref_logp"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seed_lens": seed_lens,
        "indices": indices,
        "ref_logps": ref_logps,
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
    indices = torch.zeros(batch_size, dtype=torch.long)

    for index, row in enumerate(batch):
        seq = torch.tensor(row["input_ids"], dtype=torch.long)
        seq_len = seq.numel()
        input_ids[index, :seq_len] = seq
        attention_mask[index, :seq_len] = 1
        labels[index, :seq_len] = seq
        indices[index] = int(row["index"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "indices": indices,
    }
