"""Typer sub-application for Forget Set IDentification (FSID) commands."""

from pathlib import Path

import typer
from rich.console import Console

from cleanslate.fsid.config import (
    ALLOWED_DISTRIBUTED_STRATEGIES,
    ALLOWED_FSDP_SHARDING_STRATEGIES,
    ALLOWED_RETAIN_LOSS_TYPES,
    ALLOWED_TRAIN_MODES,
    TrainerConfig,
    get_allowed_objectives,
    load_config,
)
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.pool import (
    build_candidate_pool,
    save_pool_datasets,
    split_pool_records,
)
from cleanslate.fsid.rules import (
    get_forget_set_rule,
    list_forget_set_rules,
    save_curated_sets,
)
from cleanslate.fsid.qa.cli import qa_app
from cleanslate.fsid.scoring.benchmark import compute_score, save_score
from cleanslate.fsid.training.trainer import train_unlearning
from cleanslate.fsid.validation import run_validation
from cleanslate.fsid.validation.results import (
    compare_validations,
    load_validation,
    save_validation,
)

fsid_app = typer.Typer(
    name="fsid",
    help="Forget Set IDentification: build pools, curate sets, train, and score.",
    no_args_is_help=True,
)
fsid_app.add_typer(qa_app)
console = Console()


def _run_curate(
    *,
    forget_set_rule: str,
    forget_pool_path: str,
    base_dataset: str | None,
    seed: int,
    output_dir: str | None,
    retain_dataset: str = "wikitext",
    retain_config: str = "wikitext-103-raw-v1",
    retain_split: str = "validation",
    retain_samples: int | None = None,
) -> None:
    try:
        rule = get_forget_set_rule(forget_set_rule)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--forget-set-rule") from exc
    forget_song_rows = load_local_records(forget_pool_path)
    requests, forget_rows, retain_rows = rule.curate(
        forget_song_rows,
        base_dataset=base_dataset,
        seed=seed,
        retain_dataset=retain_dataset,
        retain_config=retain_config,
        retain_split=retain_split,
        retain_samples=retain_samples,
    )
    resolved_output_dir = output_dir or str(Path("data/fsid/curated") / rule.name)
    paths = save_curated_sets(
        output_dir=resolved_output_dir,
        requests=requests,
        forget_rows=forget_rows,
        retain_rows=retain_rows,
        metadata={
            "forget_set_rule": rule.name,
            "rule_name": rule.artifact_rule_name,
            "base_dataset": base_dataset,
            "forget_pool_path": str(Path(forget_pool_path).resolve()),
            "request_count": len(requests),
            "seed": seed,
        },
    )

    console.print(f"[bold]Rule:[/bold] {rule.name}")
    console.print(f"[bold]Requests:[/bold] {len(requests)}")
    console.print(f"[bold]Forget rows:[/bold] {len(forget_rows)}")
    console.print(f"[bold]Retain rows:[/bold] {len(retain_rows)}")
    console.print(f"[bold]Saved requests:[/bold] {paths['requests']}")
    console.print(f"[bold]Saved forget set:[/bold] {paths['forget']}")
    console.print(f"[bold]Saved retain set:[/bold] {paths['retain']}")
    console.print(f"[bold]Saved metadata:[/bold] {paths['metadata']}")


@fsid_app.command("build-pool")
def build_pool(
    eval_dataset: str = typer.Option(
        ...,
        "--eval-dataset",
        help="Eval HF dataset ID (e.g. unlearning-cleanslate/eval__OLMo-3-32B__sliding_window__full)",
    ),
    fraction_col: str = typer.Option(
        "memorized_fraction",
        "--fraction-col",
        help="Column containing memorized fraction",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        "-t",
        help="Minimum memorized fraction to include in pool",
    ),
    target_size: int = typer.Option(
        100,
        "--target-size",
        help="Target pool size after stratified sampling",
    ),
    min_size: int = typer.Option(
        50,
        "--min-size",
        help="Minimum desired pool size",
    ),
    max_size: int = typer.Option(
        200,
        "--max-size",
        help="Maximum pool size",
    ),
    num_bins: int = typer.Option(
        5,
        "--num-bins",
        help="Number of memorization strata",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Sampling seed",
    ),
    deduplicate: bool = typer.Option(
        True,
        "--deduplicate/--no-deduplicate",
        help="Deduplicate candidate songs before stratified sampling",
    ),
    forget_fraction: float | None = typer.Option(
        None,
        "--forget-fraction",
        help=(
            "Fraction of pool songs to mark for unlearning, in (0, 1] "
            "(default: all pool rows)"
        ),
    ),
    output_root: str = typer.Option(
        "data/fsid/pool",
        "--output-dir",
        "--output",
        "-o",
        help="Directory or parquet root for pool/forget/retain outputs",
    ),
) -> None:
    """Build a local stratified candidate pool and split it into forget/retain songs."""
    pool_rows, stats = build_candidate_pool(
        eval_dataset=eval_dataset,
        fraction_col=fraction_col,
        threshold=threshold,
        target_size=target_size,
        min_size=min_size,
        max_size=max_size,
        num_bins=num_bins,
        seed=seed,
        deduplicate=deduplicate,
    )
    forget_rows, retain_rows = split_pool_records(
        pool_rows,
        forget_fraction=forget_fraction,
        num_bins=num_bins,
        seed=seed,
    )
    outputs = save_pool_datasets(
        pool_records=pool_rows,
        forget_records=forget_rows,
        retain_records=retain_rows,
        output_root=output_root,
    )

    console.print(f"[bold]Saved pool:[/bold] {outputs['pool']}")
    console.print(f"[bold]Saved forget songs:[/bold] {outputs['forget']}")
    console.print(f"[bold]Saved retain songs:[/bold] {outputs['retain']}")
    console.print(f"[bold]Rows scanned:[/bold] {stats.total_rows_seen}")
    console.print(f"[bold]Eligible rows:[/bold] {stats.eligible_rows}")
    console.print(f"[bold]Pool rows:[/bold] {stats.selected_rows}")
    console.print(f"[bold]Forget rows:[/bold] {len(forget_rows)}")
    console.print(f"[bold]Retain rows:[/bold] {len(retain_rows)}")
    console.print(
        f"[bold]Mem fraction:[/bold] min={stats.min_fraction:.3f} "
        f"mean={stats.mean_fraction:.3f} max={stats.max_fraction:.3f}"
    )


@fsid_app.command("curate")
def curate(
    forget_set_rule: str = typer.Option(
        "baseline",
        "--forget-set-rule",
        "--rule",
        help=(
            "Forget-set construction rule. Allowed: "
            f"{', '.join(list_forget_set_rules())}"
        ),
    ),
    forget_pool_path: str = typer.Option(
        "data/fsid/pool/forget.parquet",
        "--forget-pool-path",
        "--pool-path",
        help="Local forget-song parquet/json/jsonl path from fsid build-pool",
    ),
    base_dataset: str | None = typer.Option(
        None,
        "--base-dataset",
        help="Base dataset identifier available to forget-set rules.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Sampling seed",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=(
            "Directory for requests/forget_set/retain_set parquet files "
            "(default: data/fsid/curated/<rule>)"
        ),
    ),
    retain_dataset: str = typer.Option(
        "wikitext",
        "--retain-dataset",
        help="HF dataset for retain set (default: wikitext)",
    ),
    retain_config: str = typer.Option(
        "wikitext-103-raw-v1",
        "--retain-config",
        help="HF dataset config for retain set",
    ),
    retain_split: str = typer.Option(
        "validation",
        "--retain-split",
        help="HF dataset split for retain set",
    ),
    retain_samples: int | None = typer.Option(
        None,
        "--retain-samples",
        help="Number of retain samples to draw (default: use all after filtering)",
    ),
) -> None:
    """Curate forget/retain sets locally using a named forget-set rule."""
    _run_curate(
        forget_set_rule=forget_set_rule,
        forget_pool_path=forget_pool_path,
        base_dataset=base_dataset,
        seed=seed,
        output_dir=output_dir,
        retain_dataset=retain_dataset,
        retain_config=retain_config,
        retain_split=retain_split,
        retain_samples=retain_samples,
    )


@fsid_app.command("train")
def train(
    # YAML config (all other flags act as overrides)
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Path to YAML config file. All other flags override YAML values.",
    ),
    # Required when --config is not provided
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Base model checkpoint or HF name (required unless --config specifies it)",
    ),
    forget_path: str | None = typer.Option(
        None,
        "--forget-path",
        help="Local forget set path, parquet/json/jsonl (required unless --config specifies it)",
    ),
    # Optional overrides
    retain_path: str | None = typer.Option(
        None,
        "--retain-path",
        help="Local retain set path, parquet/json/jsonl",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for checkpoint and training summary",
    ),
    objective: str | None = typer.Option(
        None,
        "--objective",
        help=f"Unlearning objective: {', '.join(get_allowed_objectives())}",
    ),
    train_mode: str | None = typer.Option(
        None,
        "--train-mode",
        help=f"Training mode: {', '.join(ALLOWED_TRAIN_MODES)}",
    ),
    window_size: int | None = typer.Option(
        None,
        "--window-size",
        help="Prefix/suffix window size for forget pair generation",
    ),
    stride: int | None = typer.Option(
        None,
        "--stride",
        help="Window stride for forget pair generation",
    ),
    max_windows_per_song: int | None = typer.Option(
        None,
        "--max-windows-per-song",
        help="Cap forget training windows per song",
    ),
    num_epochs: int | None = typer.Option(
        None,
        "--epochs",
        help="Number of training epochs over the forget set",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Micro-batch size",
    ),
    gradient_accum_steps: int | None = typer.Option(
        None,
        "--grad-accum",
        help="Gradient accumulation steps",
    ),
    lr: float | None = typer.Option(
        None,
        "--lr",
        help="Learning rate",
    ),
    weight_decay: float | None = typer.Option(
        None,
        "--weight-decay",
        help="AdamW weight decay",
    ),
    warmup_steps: int | None = typer.Option(
        None,
        "--warmup-steps",
        help="Linear scheduler warmup steps",
    ),
    max_grad_norm: float | None = typer.Option(
        None,
        "--max-grad-norm",
        help="Gradient clipping norm (0 = disabled)",
    ),
    npo_beta: float | None = typer.Option(
        None,
        "--npo-beta",
        help="NPO beta parameter",
    ),
    simnpo_gamma: float | None = typer.Option(
        None,
        "--simnpo-gamma",
        help="SimNPO margin parameter",
    ),
    forget_weight: float | None = typer.Option(
        None,
        "--forget-weight",
        help="Forget objective weight (scales forget loss term)",
    ),
    retain_weight: float | None = typer.Option(
        None,
        "--retain-weight",
        help="Retain objective weight",
    ),
    retain_loss_type: str | None = typer.Option(
        None,
        "--retain-loss-type",
        help=f"Retain loss type: {', '.join(ALLOWED_RETAIN_LOSS_TYPES)}",
    ),
    max_seq_len: int | None = typer.Option(
        None,
        "--max-seq-len",
        help="Maximum tokenized sequence length",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Training device: auto, cuda, cpu, mps",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Training dtype: bfloat16, float16, float32",
    ),
    distributed_strategy: str | None = typer.Option(
        None,
        "--distributed-strategy",
        help=f"Distributed strategy: {', '.join(ALLOWED_DISTRIBUTED_STRATEGIES)}",
    ),
    fsdp_sharding_strategy: str | None = typer.Option(
        None,
        "--fsdp-sharding-strategy",
        help=(f"FSDP sharding strategy: {', '.join(ALLOWED_FSDP_SHARDING_STRATEGIES)}"),
    ),
    fsdp_cpu_offload: bool | None = typer.Option(
        None,
        "--fsdp-cpu-offload/--no-fsdp-cpu-offload",
        help="Enable FSDP parameter CPU offload",
    ),
    num_workers: int | None = typer.Option(
        None,
        "--num-workers",
        help="Dataloader workers",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Training seed",
    ),
    log_every: int | None = typer.Option(
        None,
        "--log-every",
        help="Log frequency in optimizer steps",
    ),
    save_every: int | None = typer.Option(
        None,
        "--save-every",
        help="Checkpoint every N steps (0 = final only)",
    ),
    eval_every: int | None = typer.Option(
        None,
        "--eval-every",
        help="Evaluate validation loss every N steps",
    ),
    eval_max_samples: int | None = typer.Option(
        None,
        "--eval-max-samples",
        help="Cap validation samples for speed",
    ),
    validation_path: str | None = typer.Option(
        None,
        "--validation-path",
        help="Local held-out validation set path (parquet/json/jsonl)",
    ),
    validation_text_col: str | None = typer.Option(
        None,
        "--validation-text-col",
        help="Text column in validation set",
    ),
    gradient_checkpointing: bool | None = typer.Option(
        None,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Enable gradient checkpointing",
    ),
    torch_compile: bool | None = typer.Option(
        None,
        "--torch-compile/--no-torch-compile",
        help="Enable torch.compile for faster training (requires use_orig_params)",
    ),
    forget_text_col: str | None = typer.Option(
        None,
        "--forget-text-col",
        help="Text column in forget set",
    ),
    retain_text_col: str | None = typer.Option(
        None,
        "--retain-text-col",
        help="Text column in retain set",
    ),
    max_forget_pairs: int | None = typer.Option(
        None,
        "--max-forget-pairs",
        help="Optional cap on forget training pairs after windowing",
    ),
    max_retain_records: int | None = typer.Option(
        None,
        "--max-retain-records",
        help="Optional cap on retain records",
    ),
    lora_r: int | None = typer.Option(
        None,
        "--lora-r",
        help="LoRA rank",
    ),
    lora_alpha: int | None = typer.Option(
        None,
        "--lora-alpha",
        help="LoRA alpha",
    ),
    lora_dropout: float | None = typer.Option(
        None,
        "--lora-dropout",
        help="LoRA dropout",
    ),
    lora_target_modules: str | None = typer.Option(
        None,
        "--lora-target-modules",
        help="Comma-separated LoRA target module names",
    ),
) -> None:
    """
    Run local PyTorch unlearning training.

    Usage with YAML config (recommended for paper experiments):

      cleanslate fsid train --config cleanslate/fsid/configs/npo.yaml -m allenai/OLMo-3-32B

    CLI flags override YAML values when both are specified.

    Usage without YAML (quick ad-hoc runs):

      cleanslate fsid train -m allenai/OLMo-3-32B --forget-path data/fsid/curated/baseline/forget_set.parquet
    """
    # Collect CLI overrides — only non-None values override YAML.
    cli_overrides: dict[str, object] = {
        "model_name": model,
        "forget_path": forget_path,
        "retain_path": retain_path,
        "output_dir": output_dir,
        "objective": objective,
        "train_mode": train_mode,
        "window_size": window_size,
        "stride": stride,
        "max_windows_per_song": max_windows_per_song,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accum_steps": gradient_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "npo_beta": npo_beta,
        "simnpo_gamma": simnpo_gamma,
        "forget_weight": forget_weight,
        "retain_weight": retain_weight,
        "retain_loss_type": retain_loss_type,
        "max_seq_len": max_seq_len,
        "device": device,
        "dtype": dtype,
        "distributed_strategy": distributed_strategy,
        "fsdp_sharding_strategy": fsdp_sharding_strategy,
        "fsdp_cpu_offload": fsdp_cpu_offload,
        "num_workers": num_workers,
        "seed": seed,
        "log_every": log_every,
        "save_every": save_every,
        "eval_every": eval_every,
        "eval_max_samples": eval_max_samples,
        "validation_path": validation_path,
        "validation_text_col": validation_text_col,
        "gradient_checkpointing": gradient_checkpointing,
        "torch_compile": torch_compile,
        "forget_text_col": forget_text_col,
        "retain_text_col": retain_text_col,
        "max_forget_pairs": max_forget_pairs,
        "max_retain_records": max_retain_records,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
    }

    if config_path is not None:
        config = load_config(config_path, cli_overrides=cli_overrides)
    else:
        # No YAML — build config directly from CLI flags.
        # model_name and forget_path are required in this mode.
        if model is None:
            console.print(
                "[red]Error: --model is required when --config is not provided.[/red]"
            )
            raise typer.Exit(1)
        if forget_path is None:
            console.print(
                "[red]Error: --forget-path is required when --config is not provided.[/red]"
            )
            raise typer.Exit(1)

        # Filter out None values so dataclass defaults apply.
        kwargs = {k: v for k, v in cli_overrides.items() if v is not None}
        config = TrainerConfig(**kwargs)  # type: ignore[arg-type]
        config.validate()

    summary = train_unlearning(config)
    if not summary.get("is_main_process", True):
        return
    console.print(f"[bold]Checkpoint:[/bold] {summary['checkpoint_dir']}")
    console.print(f"[bold]Run time:[/bold] {summary['run_seconds']:.1f}s")
    console.print(f"[bold]Forget rows:[/bold] {summary['num_forget_rows']}")
    console.print(f"[bold]Forget pairs:[/bold] {summary['num_forget_pairs']}")
    console.print(f"[bold]Retain rows:[/bold] {summary['num_retain_rows']}")
    if summary.get("num_val_sequences", 0) > 0:
        console.print(f"[bold]Val sequences:[/bold] {summary['num_val_sequences']}")


@fsid_app.command("validate")
def validate(
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        help="Model name or checkpoint path",
    ),
    benchmarks: str = typer.Option(
        "mmlu,gsm8k",
        "-B",
        "--benchmarks",
        help="Comma-separated lm-eval task names (e.g. mmlu,gsm8k,arc_easy,hellaswag)",
    ),
    output: str = typer.Option(
        "results/fsid/validation.json",
        "-o",
        "--output",
        help="Output JSON path",
    ),
    limit: int | None = typer.Option(
        None,
        "-n",
        "--limit",
        help="Limit items per task (for quick testing)",
    ),
    num_fewshot: int | None = typer.Option(
        None,
        "--num-fewshot",
        help="Override default number of few-shot examples",
    ),
    batch_size: str = typer.Option(
        "auto",
        "--batch-size",
        help="Batch size for evaluation ('auto' or integer)",
    ),
    mem_fraction: float = typer.Option(
        0.65,
        "--mem-fraction",
        help="GPU memory fraction for SGLang engine",
    ),
    tp_size: int = typer.Option(
        1,
        "--tp-size",
        help="Tensor parallel size for multi-GPU",
    ),
    num_gpus: int = typer.Option(
        1,
        "--num-gpus",
        "-g",
        help="Number of GPUs (split across TP and DP automatically)",
    ),
    gpu_ids: str | None = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU IDs to use (e.g. '2,3,4,5'). Default: 0 to num_gpus-1",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
) -> None:
    """Run validation benchmarks on a model using lm-evaluation-harness."""
    import os

    task_names = [b.strip() for b in benchmarks.split(",") if b.strip()]

    # SGLang supports TP + DP.  Compute dp_size from num_gpus and tp_size.
    if num_gpus % tp_size != 0:
        console.print(
            f"[red]Error: --num-gpus ({num_gpus}) must be divisible by --tp-size ({tp_size})[/red]"
        )
        raise typer.Exit(1)
    dp_size = num_gpus // tp_size

    # Set CUDA_VISIBLE_DEVICES for multi-GPU
    if num_gpus > 1:
        if gpu_ids:
            gpus = [int(g.strip()) for g in gpu_ids.split(",")]
            if len(gpus) != num_gpus:
                console.print(
                    f"[red]Error: --gpu-ids has {len(gpus)} GPUs but --num-gpus is {num_gpus}[/red]"
                )
                raise typer.Exit(1)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(g) for g in range(num_gpus)
            )

    # Parse batch_size: allow "auto" or integer string
    bs: int | str
    try:
        bs = int(batch_size)
    except ValueError:
        bs = batch_size

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Tasks:[/bold] {', '.join(task_names)}")
    if tp_size > 1:
        console.print(f"[bold]TP size:[/bold] {tp_size}")
    if dp_size > 1:
        console.print(f"[bold]DP size:[/bold] {dp_size}")
    if limit is not None:
        console.print(f"[bold]Limit:[/bold] {limit} items per task")

    results = run_validation(
        model_name=model,
        tasks=task_names,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=bs,
        mem_fraction=mem_fraction,
        tp_size=tp_size,
        dp_size=dp_size,
    )

    out = save_validation(results, output, model_name=model)

    console.print()
    for name, result in results.items():
        console.print(f"[bold]{name.upper()}[/bold]")
        console.print(f"  Accuracy: {result.accuracy:.1%}")
        console.print(f"  Time: {result.eval_time_seconds:.1f}s")
        if result.per_subset:
            top_subjects = sorted(
                result.per_subset.values(), key=lambda s: s.accuracy, reverse=True
            )[:5]
            console.print(
                f"  Top subjects: {', '.join(s.name + f' ({s.accuracy:.0%})' for s in top_subjects)}"
            )
        console.print()

    console.print(f"[bold]Saved to:[/bold] {out}")


@fsid_app.command("score")
def score(
    forget_set_path: str = typer.Option(
        ...,
        "--forget-path",
        help="Local forget_set.parquet produced by fsid curate",
    ),
    pre_eval_path: str = typer.Option(
        ...,
        "--pre-eval",
        help="Eval results BEFORE unlearning: local parquet path or HF dataset ID (org/dataset or org/dataset:config)",
    ),
    post_eval_path: str = typer.Option(
        ...,
        "--post-eval",
        help="Eval results AFTER unlearning: local parquet path or HF dataset ID (org/dataset or org/dataset:config)",
    ),
    pre_validation_path: str | None = typer.Option(
        None,
        "--pre-validation",
        help="Pre-unlearning validation JSON (from fsid validate)",
    ),
    post_validation_path: str | None = typer.Option(
        None,
        "--post-validation",
        help="Post-unlearning validation JSON (from fsid validate)",
    ),
    output_path: str = typer.Option(
        "results/fsid/score.json",
        "--output",
        "-o",
        help="Output JSON path for score report",
    ),
) -> None:
    """
    Score an unlearning run by comparing pre/post memorization metrics.

    Forget score: fraction of forget-set songs with reduced memorization (higher is better).
    Retain penalty: normalized change in memorization for non-forget songs (lower is better).
    Combined score: forget_score - retain_weight * retain_penalty.

    Optionally include --pre-validation and --post-validation to measure utility retention.
    """
    result = compute_score(
        forget_set_path=forget_set_path,
        pre_eval_path=pre_eval_path,
        post_eval_path=post_eval_path,
    )
    out = save_score(result, output_path)

    console.print()
    console.print("[bold]Forget set[/bold]")
    console.print(f"  Songs scored:           {result.forget_count}")
    console.print(f"  Fraction improved:      {result.forget_fraction_improved:.1%}")
    console.print(
        f"  Mean Δ memorized_frac:  {result.forget_mean_delta_frac:+.4f}  (positive = less memorized)"
    )
    console.print(f"  Mean Δ max_p_z:         {result.forget_mean_delta_pz:+.6f}")
    if result.forget_mean_delta_mean_ppl is not None:
        console.print(
            f"  Mean Δ mean_ppl:        {result.forget_mean_delta_mean_ppl:+.4f}  (positive = higher ppl, more forgotten)"
        )
    if result.forget_mean_delta_min_ppl is not None:
        console.print(
            f"  Mean Δ min_ppl:         {result.forget_mean_delta_min_ppl:+.4f}"
        )
    if result.forget_total_windows is not None:
        console.print(f"  Total windows:          {result.forget_total_windows}")
        console.print(
            f"  Windows forgotten:      {result.forget_windows_forgotten}  (memorized -> clean)"
        )
        console.print(
            f"  Windows regressed:      {result.forget_windows_regressed}  (clean -> memorized)"
        )
        console.print(
            f"  Window forget rate:     {result.forget_window_forget_rate:.1%}"
        )
    console.print()
    console.print("[bold]Retain set[/bold]")
    console.print(f"  Songs scored:           {result.retain_count}")
    console.print(
        f"  Mean |Δ| memorized_frac:{result.retain_mean_abs_delta_frac:.4f}  (lower = better)"
    )
    console.print(f"  Mean |Δ| max_p_z:       {result.retain_mean_abs_delta_pz:.6f}")
    if result.retain_mean_abs_delta_mean_ppl is not None:
        console.print(
            f"  Mean |Δ| mean_ppl:      {result.retain_mean_abs_delta_mean_ppl:.4f}  (lower = better)"
        )
    if result.retain_mean_abs_delta_min_ppl is not None:
        console.print(
            f"  Mean |Δ| min_ppl:       {result.retain_mean_abs_delta_min_ppl:.4f}"
        )
    if result.retain_total_windows is not None:
        console.print(f"  Total windows:          {result.retain_total_windows}")
        console.print(
            f"  Windows regressed:      {result.retain_windows_regressed}  (clean -> memorized)"
        )
        console.print(f"  Window stability:       {result.retain_window_stability:.1%}")
    console.print()
    console.print("[bold]Score[/bold]")
    console.print(f"  Forget score:    [green]{result.forget_score:.4f}[/green]")
    console.print(f"  Retain penalty:  [yellow]{result.retain_penalty:.4f}[/yellow]")

    # Validation comparison
    if pre_validation_path and post_validation_path:
        pre_val = load_validation(pre_validation_path)
        post_val = load_validation(post_validation_path)
        deltas = compare_validations(pre_val, post_val)
        if deltas:
            console.print()
            console.print("[bold]Validation[/bold]")
            for d in deltas:
                console.print(
                    f"  {d.benchmark_name.upper():8s}"
                    f"  pre={d.pre_accuracy:.1%}"
                    f"  post={d.post_accuracy:.1%}"
                    f"  Δ={d.delta_accuracy:+.1%}"
                    f"  ({d.relative_retention:.1%} retained)"
                )
            avg_retention = sum(d.relative_retention for d in deltas) / len(deltas)
            console.print(f"  Utility retained: {avg_retention:.1%}")

    console.print()
    console.print(f"[bold]Saved to:[/bold] {out}")
