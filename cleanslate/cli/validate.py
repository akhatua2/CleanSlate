"""cleanslate validate -- run lm-evaluation-harness benchmarks."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cleanslate.cli.aliases import resolve_model

console = Console()

DEFAULT_TASKS = "mmlu,hellaswag,arc_easy,arc_challenge"


def validate_cmd(
    model: str = typer.Option(..., "-m", "--model", help="HF model name or alias"),
    tasks: str = typer.Option(DEFAULT_TASKS, "-B", "--benchmarks", help="Comma-separated lm-eval tasks"),
    limit: int | None = typer.Option(None, "-n", "--limit", help="Max items per task (for quick testing)"),
    num_fewshot: int | None = typer.Option(None, "--num-fewshot"),
    batch_size: str = typer.Option("auto", "--batch-size"),
    gpu_memory_utilization: float = typer.Option(0.90, "--gpu-mem"),
    tp_size: int = typer.Option(1, "--tp-size", help="Tensor parallel size"),
    max_model_len: int = typer.Option(4096, "--max-model-len"),
    base_url: str | None = typer.Option(None, "--base-url", help="Remote vLLM OpenAI endpoint (e.g. https://xxx.modal.run)"),
    num_concurrent: int = typer.Option(32, "--num-concurrent", help="In-flight requests vs remote endpoint"),
    output: str | None = typer.Option(None, "-o", "--output", help="Path to save JSON results"),
    compare: str | None = typer.Option(None, "--compare", help="Path to previous validation JSON for comparison"),
) -> None:
    """Run lm-evaluation-harness benchmarks via vLLM."""
    from cleanslate.validation import run_validation, save_validation, load_validation, compare_validations

    model_path, model_clean = resolve_model(model)
    task_list = [t.strip() for t in tasks.split(",")]

    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Tasks:[/bold] {', '.join(task_list)}")
    if base_url:
        console.print(f"[bold]Remote endpoint:[/bold] {base_url}")
    if limit:
        console.print(f"[bold]Limit:[/bold] {limit} items per task")
    console.print()

    results = run_validation(
        model_name=model_path,
        tasks=task_list,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        tp_size=tp_size,
        max_model_len=max_model_len,
        base_url=base_url,
        num_concurrent=num_concurrent,
    )

    # Display results table
    table = Table(title="Validation Results")
    table.add_column("Benchmark", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Time (s)", justify="right")

    for name, result in sorted(results.items()):
        table.add_row(name, f"{result.accuracy:.4f}", f"{result.eval_time_seconds:.1f}")
    console.print(table)

    # Save
    if output:
        out_path = save_validation(results, output, model_path)
        console.print(f"\n[green]Saved to {out_path}[/green]")
    else:
        default_out = Path("results/validation") / f"{model_clean}.json"
        out_path = save_validation(results, default_out, model_path)
        console.print(f"\n[green]Saved to {out_path}[/green]")

    # Compare with previous run
    if compare:
        pre = load_validation(compare)
        post = {"benchmarks": {k: v.to_dict() for k, v in results.items()}}
        deltas = compare_validations(pre, post)

        if deltas:
            comp_table = Table(title="Comparison with previous run")
            comp_table.add_column("Benchmark", style="bold")
            comp_table.add_column("Pre", justify="right")
            comp_table.add_column("Post", justify="right")
            comp_table.add_column("Delta", justify="right")
            comp_table.add_column("Retention", justify="right")

            for d in deltas:
                delta_style = "green" if d.delta_accuracy >= 0 else "red"
                comp_table.add_row(
                    d.benchmark_name,
                    f"{d.pre_accuracy:.4f}",
                    f"{d.post_accuracy:.4f}",
                    f"[{delta_style}]{d.delta_accuracy:+.4f}[/{delta_style}]",
                    f"{d.relative_retention:.2%}",
                )
            console.print(comp_table)
