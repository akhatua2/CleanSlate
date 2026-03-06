"""Data utilities for FSID unlearning with HF Trainer, ported from open-unlearning."""

from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset


class ForgetRetainDataset(TorchDataset):
    """Pairs forget + retain samples, anchoring on forget set length.

    Follows open-unlearning's ForgetRetainDataset pattern: each __getitem__
    returns a dict with "forget" and "retain" sub-dicts. Retain indices
    wrap around if the retain set is smaller than the forget set.
    """

    def __init__(
        self,
        forget_dataset: TorchDataset,
        retain_dataset: TorchDataset | None = None,
    ) -> None:
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset

    def __len__(self) -> int:
        return len(self.forget_dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {"forget": self.forget_dataset[index]}
        if self.retain_dataset is not None and len(self.retain_dataset) > 0:  # type: ignore[arg-type]
            retain_index = index % len(self.retain_dataset)  # type: ignore[arg-type]
            result["retain"] = self.retain_dataset[retain_index]
        return result


class UnlearnDataCollator:
    """Recursive collator that handles nested forget/retain dicts.

    Each sub-dict is padded independently using standard HF-style
    right-padding for input_ids, attention_mask, and labels.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _collate_subdict(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, row in enumerate(batch):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long) if not isinstance(ids, torch.Tensor) else ids
            attention_mask[i, :seq_len] = 1
            row_labels = row["labels"]
            labels[i, :seq_len] = torch.tensor(row_labels, dtype=torch.long) if not isinstance(row_labels, torch.Tensor) else row_labels

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["forget"] = self._collate_subdict([item["forget"] for item in batch])
        if "retain" in batch[0]:
            result["retain"] = self._collate_subdict([item["retain"] for item in batch])
        return result
