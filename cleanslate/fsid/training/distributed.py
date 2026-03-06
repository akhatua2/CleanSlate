"""Distributed training utilities for FSID."""

import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from cleanslate.fsid.config import TrainerConfig

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


def init_distributed(config: TrainerConfig) -> DistributedContext:
    """Initialize the distributed runtime if requested."""
    if config.distributed_strategy != "fsdp":
        return DistributedContext(
            strategy="single",
            rank=0,
            world_size=1,
            local_rank=0,
            is_distributed=False,
            is_main_process=True,
            device="",
        )

    if not torch.cuda.is_available():
        raise ValueError("FSDP requires CUDA, but CUDA is unavailable")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if world_size < 2 or rank < 0 or local_rank < 0:
        raise ValueError(
            "FSDP requires torchrun-style launch metadata (WORLD_SIZE, RANK, LOCAL_RANK) "
            "with WORLD_SIZE >= 2"
        )

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    return DistributedContext(
        strategy="fsdp",
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=True,
        is_main_process=(rank == 0),
        device=f"cuda:{local_rank}",
    )


def destroy_distributed(context: DistributedContext) -> None:
    """Destroy the distributed process group if active."""
    if context.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


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


def reduce_scalar_mean(
    value: float,
    *,
    device: str,
    context: DistributedContext,
) -> float:
    """Average a scalar across ranks."""
    if not context.is_distributed:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= context.world_size
    return float(tensor.item())


def reduce_scalar_sum(
    value: float,
    *,
    device: str,
    context: DistributedContext,
) -> float:
    """Sum a scalar across ranks."""
    if not context.is_distributed:
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


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

    gathered: list[dict[int, float] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_values)

    output: list[float | None] = [None] * dataset_len
    for shard in gathered:
        if shard is None:
            raise RuntimeError("all_gather_object returned a missing shard")
        for index, value in shard.items():
            if output[index] is not None:
                raise RuntimeError(f"Reference precompute duplicated index {index}")
            output[index] = value

    if any(value is None for value in output):
        raise RuntimeError("Reference precompute did not cover the full dataset")
    return [float(value) for value in output]


def wrap_model_for_fsdp(
    model: Any,
    *,
    config: TrainerConfig,
    torch_dtype: torch.dtype,
) -> Any:
    """Wrap a single-device model with FSDP."""
    layer = _resolve_transformer_layer(model, 0)
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(layer)},
    )
    mixed_precision = None
    if torch_dtype != torch.float32:
        mixed_precision = MixedPrecision(
            param_dtype=torch_dtype,
            reduce_dtype=torch_dtype,
            buffer_dtype=torch_dtype,
        )

    sharding_strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }[config.fsdp_sharding_strategy]

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=config.fsdp_cpu_offload),
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        device_id=torch.cuda.current_device(),
        sync_module_states=False,
        use_orig_params=False,
    )


def save_model_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    path: Path,
    context: DistributedContext,
    train_mode: str,
) -> None:
    """Save a checkpoint for either single-device or FSDP training."""
    path.mkdir(parents=True, exist_ok=True)

    if not context.is_distributed:
        model.save_pretrained(str(path))
        tokenizer.save_pretrained(str(path))
        return

    if train_mode != "full":
        raise ValueError("FSDP checkpoint export only supports full fine-tuning")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        full_state_dict = model.state_dict()

    if context.is_main_process:
        model.module.save_pretrained(str(path), state_dict=full_state_dict)
        tokenizer.save_pretrained(str(path))
    dist.barrier()
