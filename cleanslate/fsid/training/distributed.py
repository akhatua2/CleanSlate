"""Distributed training utilities for FSID."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

_LAYER_PATHS = [
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "decoder.layers",
]


def _resolve_transformer_layer(model: Any, layer_id: int) -> Any:
    """Resolve a transformer block from a causal LM for FSDP wrapping."""
    base = model
    while hasattr(base, "base_model"):
        base = base.base_model
    while hasattr(base, "model") and not hasattr(base, "lm_head"):
        base = base.model

    for path in _LAYER_PATHS:
        obj: Any = base
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if not ok or obj is None:
            continue
        try:
            return obj[layer_id]
        except IndexError as exc:
            raise IndexError(
                f"layer_id={layer_id} is out of range for '{path}' with "
                f"{len(obj)} layers."
            ) from exc

    raise AttributeError(
        f"Cannot find transformer layers on {type(model).__name__}. "
        f"Tried: {_LAYER_PATHS}."
    )


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed-training context."""

    strategy: str
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    is_main_process: bool
    device: str


def partition_dataset(dataset: Dataset, context: DistributedContext) -> Dataset:
    """Return a disjoint local partition for eval-style passes."""
    if not context.is_distributed:
        return dataset
    indices = list(range(context.rank, len(dataset), context.world_size))
    return Subset(dataset, indices)


def build_training_sampler(
    dataset: Dataset,
    *,
    shuffle: bool,
    seed: int,
    context: DistributedContext,
) -> DistributedSampler | None:
    """Build a distributed sampler for training datasets."""
    if not context.is_distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def gather_indexed_scalars(
    local_values: dict[int, float],
    *,
    dataset_len: int,
    context: DistributedContext,
) -> list[float]:
    """Gather per-index scalar values from all ranks and rebuild the full list."""
    if not context.is_distributed:
        output = [0.0] * dataset_len
        for index, value in local_values.items():
            output[index] = value
        return output

    # Use a dense tensor + all_reduce(SUM) instead of all_gather_object.
    # all_gather_object serializes Python dicts to byte tensors on the CUDA
    # device via NCCL; if any rank reports an unexpected serialized size the
    # receiving ranks try to resize to that value and OOM instantly.
    # Because partition_dataset assigns non-overlapping indices to each rank,
    # summing a zero-initialised tensor gives the correct merged result.
    tensor = torch.zeros(dataset_len, dtype=torch.float64)
    for index, value in local_values.items():
        tensor[index] = value
    tensor = tensor.to(context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()
