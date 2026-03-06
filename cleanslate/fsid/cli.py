"""Typer sub-application for Forget Set IDentification (FSID) commands."""

from pathlib import Path

import typer
from rich.console import Console

from cleanslate.fsid.config import (
    ALLOWED_DISTRIBUTED_STRATEGIES,
    ALLOWED_FSDP_SHARDING_STRATEGIES,
    ALLOWED_TRAIN_MODES,
    TrainerConfig,
    get_allowed_objectives,
    load_config,
)
from cleanslate.fsid.datasets.io import load_local_records
from cleanslate.fsid.datasets.pool import build_candidate_pool, save_pool_dataset
from cleanslate.fsid.rules.baseline import (
    curate_baseline_sets,
    save_curated_sets,
)
from cleanslate.fsid.scoring.benchmark import compute_score, save_score
from cleanslate.fsid.training.trainer import train_unlearning

fsid_app = typer.Typer(
    name="fsid",
    help="Forget Set IDentification: build pools, curate sets, train, and score.",
    no_args_is_help=True,
)
console = Console()


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
    output_path: str = typer.Option(
        "data/fsid/pool.parquet",
        "--output",
        "-o",
        help="Local parquet output path for candidate pool",
    ),
) -> None:
    """Build a local stratified candidate pool from an eval dataset."""
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
    output = save_pool_dataset(pool_rows, output_path)

    console.print(f"[bold]Saved pool:[/bold] {output}")
    console.print(f"[bold]Rows scanned:[/bold] {stats.total_rows_seen}")
    console.print(f"[bold]Eligible rows:[/bold] {stats.eligible_rows}")
    console.print(f"[bold]Selected rows:[/bold] {stats.selected_rows}")
    console.print(
        f"[bold]Mem fraction:[/bold] min={stats.min_fraction:.3f} "
        f"mean={stats.mean_fraction:.3f} max={stats.max_fraction:.3f}"
    )


@fsid_app.command("curate-baseline")
def curate_baseline(
    pool_path: str = typer.Option(
        "data/fsid/pool.parquet",
        "--pool-path",
        help="Local pool parquet/json/jsonl path",
    ),
    request_count: int | None = typer.Option(
        None,
        "--request-count",
        "-n",
        help="Number of requests to curate (default: all pool rows)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Sampling seed",
    ),
    num_bins: int = typer.Option(
        5,
        "--num-bins",
        help="Number of memorization strata for request selection",
    ),
    output_dir: str = typer.Option(
        "data/fsid/curated/baseline",
        "--output-dir",
        "-o",
        help="Directory for requests/forget_set/retain_set parquet files",
    ),
) -> None:
    """
    Curate baseline forget/retain sets locally.

    Baseline policy:
      - D_forget = {(song, full lyrics)}
      - D_retain = empty
    """
    pool_rows = load_local_records(pool_path)
    requests, forget_rows, retain_rows = curate_baseline_sets(
        pool_rows,
        request_count=request_count,
        seed=seed,
        num_bins=num_bins,
    )
    paths = save_curated_sets(
        output_dir=output_dir,
        requests=requests,
        forget_rows=forget_rows,
        retain_rows=retain_rows,
        metadata={
            "rule_name": "baseline_literal",
            "pool_path": str(Path(pool_path).resolve()),
            "request_count": len(requests),
            "seed": seed,
            "num_bins": num_bins,
        },
    )

    console.print(f"[bold]Requests:[/bold] {len(requests)}")
    console.print(f"[bold]Forget rows:[/bold] {len(forget_rows)}")
    console.print(f"[bold]Retain rows:[/bold] {len(retain_rows)}")
    console.print(f"[bold]Saved requests:[/bold] {paths['requests']}")
    console.print(f"[bold]Saved forget set:[/bold] {paths['forget']}")
    console.print(f"[bold]Saved retain set:[/bold] {paths['retain']}")
    console.print(f"[bold]Saved metadata:[/bold] {paths['metadata']}")


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
    base_dataset: str | None = typer.Option(
        None,
        "--base-dataset",
        help="Training corpus the model was trained on (default: allenai/dolma3_mix-6T)",
    ),
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
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Number of optimizer steps",
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
    retain_weight: float | None = typer.Option(
        None,
        "--retain-weight",
        help="Retain objective weight",
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
        help=(
            "FSDP sharding strategy: "
            f"{', '.join(ALLOWED_FSDP_SHARDING_STRATEGIES)}"
        ),
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
        "base_dataset": base_dataset,
        "retain_path": retain_path,
        "output_dir": output_dir,
        "objective": objective,
        "train_mode": train_mode,
        "window_size": window_size,
        "stride": stride,
        "max_windows_per_song": max_windows_per_song,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accum_steps": gradient_accum_steps,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "npo_beta": npo_beta,
        "simnpo_gamma": simnpo_gamma,
        "retain_weight": retain_weight,
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


@fsid_app.command("score")
def score(
    forget_set_path: str = typer.Option(
        ...,
        "--forget-path",
        help="Local forget_set.parquet from fsid curate-baseline",
    ),
    pre_eval_path: str = typer.Option(
        ...,
        "--pre-eval",
        help="Local eval results parquet BEFORE unlearning",
    ),
    post_eval_path: str = typer.Option(
        ...,
        "--post-eval",
        help="Local eval results parquet AFTER unlearning (run cleanslate eval on checkpoint)",
    ),
    retain_weight: float = typer.Option(
        1.0,
        "--retain-weight",
        help="Weight for retain penalty in combined score",
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
    """
    result = compute_score(
        forget_set_path=forget_set_path,
        pre_eval_path=pre_eval_path,
        post_eval_path=post_eval_path,
        retain_weight=retain_weight,
    )
    out = save_score(result, output_path)

    console.print()
    console.print("[bold]Forget set[/bold]")
    console.print(f"  Songs scored:           {result.forget_count}")
    console.print(f"  Fraction improved:      {result.forget_fraction_improved:.1%}")
    console.print(f"  Mean Δ memorized_frac:  {result.forget_mean_delta_frac:+.4f}  (positive = less memorized)")
    console.print(f"  Mean Δ max_p_z:         {result.forget_mean_delta_pz:+.6f}")
    console.print()
    console.print("[bold]Retain set[/bold]")
    console.print(f"  Songs scored:           {result.retain_count}")
    console.print(f"  Mean |Δ| memorized_frac:{result.retain_mean_abs_delta_frac:.4f}  (lower = better)")
    console.print(f"  Mean |Δ| max_p_z:       {result.retain_mean_abs_delta_pz:.6f}")
    console.print()
    console.print("[bold]Score[/bold]")
    console.print(f"  Forget score:    [green]{result.forget_score:.4f}[/green]")
    console.print(f"  Retain penalty:  [yellow]{result.retain_penalty:.4f}[/yellow]")
    console.print(f"  Combined score:  [bold]{result.combined_score:.4f}[/bold]  "
                  f"(forget - {retain_weight}×retain_penalty)")
    console.print()
    console.print(f"[bold]Saved to:[/bold] {out}")
