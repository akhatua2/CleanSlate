"""Paired forget/retain datasets and nested collator for unlearning."""

from typing import Any, Literal

import torch
from torch.utils.data import Dataset as TorchDataset

type Anchor = Literal["forget", "retain"]


class ForgetRetainDataset(TorchDataset[dict[str, Any]]):
    """Pair a forget sample with a random retain sample (or vice versa) per item.

    Open-unlearning's ForgetRetainDataset pattern: anchor one side, sample the other.
    Random pairing avoids deterministic retain selection that can overfit regularization.
    """

    def __init__(
        self,
        forget_dataset: TorchDataset[dict[str, Any]],
        retain_dataset: TorchDataset[dict[str, Any]] | None = None,
        anchor: Anchor = "forget",
    ) -> None:
        if anchor == "retain" and retain_dataset is None:
            raise ValueError("anchor='retain' requires a retain_dataset")
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.anchor = anchor

    def _len(self, ds: TorchDataset[dict[str, Any]]) -> int:
        return len(ds)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self._len(self.retain_dataset) if self.anchor == "retain" and self.retain_dataset else self._len(self.forget_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        match self.anchor:
            case "forget":
                result = {"forget": self.forget_dataset[index]}
                if self.retain_dataset is not None and self._len(self.retain_dataset) > 0:
                    ri = int(torch.randint(0, self._len(self.retain_dataset), (1,)).item())
                    result["retain"] = self.retain_dataset[ri]
                return result
            case "retain":
                assert self.retain_dataset is not None  # guaranteed by __init__
                result = {"retain": self.retain_dataset[index]}
                if self._len(self.forget_dataset) > 0:
                    fi = int(torch.randint(0, self._len(self.forget_dataset), (1,)).item())
                    result["forget"] = self.forget_dataset[fi]
                return result


class UnlearnDataCollator:
    """Recursive collator: pads forget/retain sub-dicts independently with right-padding."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        for i, row in enumerate(batch):
            ids = row["input_ids"]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(row["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"forget": self._collate([item["forget"] for item in batch])}
        if "retain" in batch[0]:
            result["retain"] = self._collate([item["retain"] for item in batch])
        return result
