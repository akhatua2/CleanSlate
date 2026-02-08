"""Command-line interface for cleanslate."""

from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install cleanslate[cli]"
    )

from .experiment import Experiment
from .generation import GenerationExperiment
from .results import ExperimentResults
from .samples import SHAKESPEARE_SAMPLES, BOOK_SAMPLES, SONG_SAMPLES

app = typer.Typer(
    name="cleanslate",
    help="Sequence likelihood estimation for LLMs - measure memorization and token probabilities.",
)
console = Console()


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    dataset: str = typer.Option(
        "shakespeare", "--dataset", "-d", help="Dataset: shakespeare, books, or all"
    ),
    name: str = typer.Option("experiment", "--name", "-n", help="Experiment name"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
):
    """Run sequence likelihood experiment on a dataset."""

    # Select samples
    if dataset == "shakespeare":
        samples = SHAKESPEARE_SAMPLES
    elif dataset == "books":
        samples = BOOK_SAMPLES
    elif dataset == "all":
        samples = SHAKESPEARE_SAMPLES + BOOK_SAMPLES
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print("Available: shakespeare, books, all")
        raise typer.Exit(1)

    console.print(f"[bold]Running experiment:[/bold] {name}")
    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Dataset:[/bold] {dataset} ({len(samples)} samples)")
    console.print()

    # Run experiment
    exp = Experiment(model=model, name=name, device=device)
    results = exp.run(samples)

    # Display results
    console.print()
    console.print(results.summary())

    # Save if output specified
    if output:
        results.save(output)


@app.command()
def probe(
    seed: str = typer.Argument(..., help="Seed/prompt text"),
    target: str = typer.Argument(..., help="Target continuation text"),
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps"
    ),
):
    """Probe a single seed/target pair."""

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Seed:[/bold] {seed[:50]}...")
    console.print(f"[bold]Target:[/bold] {target[:50]}...")
    console.print()

    exp = Experiment(model=model, name="probe", device=device)
    result = exp.run_single(seed, target)

    # Display result
    table = Table(title="Probe Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Log-Likelihood", f"{result.log_likelihood:.2f}")
    table.add_row("Perplexity", f"{result.perplexity:.2f}")
    table.add_row("Tokens", str(result.num_tokens))
    table.add_row("Rank-1 Fraction", f"{result.rank_1_fraction:.1%}")
    table.add_row("Avg Rank", f"{result.avg_rank:.1f}")

    console.print(table)


@app.command()
def compare(
    file1: str = typer.Argument(..., help="First results JSON file"),
    file2: str = typer.Argument(..., help="Second results JSON file"),
):
    """Compare two experiment results."""

    results1 = ExperimentResults.load(file1)
    results2 = ExperimentResults.load(file2)

    console.print(results1.compare(results2))


@app.command()
def show(
    file: str = typer.Argument(..., help="Results JSON file to display"),
    tokens: bool = typer.Option(False, "--tokens", "-t", help="Show token counts"),
):
    """Display saved experiment results."""

    results = ExperimentResults.load(file)
    console.print(results.summary(show_tokens=tokens))


@app.command()
def list_samples():
    """List available sample datasets."""

    console.print("[bold]Shakespeare Samples:[/bold]")
    for i, s in enumerate(SHAKESPEARE_SAMPLES, 1):
        console.print(f"  {i}. {s['name']}")

    console.print()
    console.print("[bold]Book Samples:[/bold]")
    for i, s in enumerate(BOOK_SAMPLES, 1):
        console.print(f"  {i}. {s['name']}")


@app.command()
def generate(
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct", "--model", "-m", help="Model name"
    ),
    dataset: str = typer.Option(
        "all",
        "--dataset",
        "-d",
        help="Dataset: shakespeare, books, songs, or all",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path (.parquet)",
    ),
    backend: str = typer.Option(
        "huggingface",
        "--backend",
        "-b",
        help="Backend: huggingface or sglang",
    ),
    num_generations: int = typer.Option(
        3, "--num-generations", "-n", help="Number of generations per seed"
    ),
    max_tokens: int = typer.Option(
        128, "--max-tokens", help="Maximum tokens per generation"
    ),
    temperature: float = typer.Option(
        1.0, "--temperature", "-t", help="Sampling temperature"
    ),
    top_p: Optional[float] = typer.Option(
        None, "--top-p", help="Nucleus sampling (top-p)"
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k sampling"),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for generation"),
    hf_repo: Optional[str] = typer.Option(
        None,
        "--hf-repo",
        help="Push to HuggingFace Hub repo (e.g. unlearning-cleanslate/generations)",
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
):
    """
    Generate multiple outputs per seed and save all data (no metrics computed).

    Saves: tokens, logprobs, top-k alternatives, full generated text, etc.
    Use --backend sglang for GPU-optimized batch generation.
    """

    # Select samples
    if dataset == "shakespeare":
        samples = SHAKESPEARE_SAMPLES
    elif dataset == "books":
        samples = BOOK_SAMPLES
    elif dataset == "songs":
        samples = SONG_SAMPLES
    elif dataset == "all":
        samples = SHAKESPEARE_SAMPLES + BOOK_SAMPLES + SONG_SAMPLES
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print("Available: shakespeare, books, songs, all")
        raise typer.Exit(1)

    console.print(f"[bold]Model:[/bold] {model}")
    console.print(f"[bold]Backend:[/bold] {backend}")
    console.print(f"[bold]Dataset:[/bold] {dataset} ({len(samples)} samples)")
    console.print(
        f"[bold]Generations:[/bold] {num_generations} x {len(samples)} = "
        f"{num_generations * len(samples)} total"
    )
    console.print(f"[bold]Batch size:[/bold] {batch_size}")
    console.print()

    exp = GenerationExperiment(model=model, backend=backend, device=device)
    rows = exp.run(
        samples=samples,
        dataset_name=dataset,
        output_path=output,
        num_generations=num_generations,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        batch_size=batch_size,
        hf_repo=hf_repo,
    )

    console.print()
    console.print(f"[green]Saved {len(rows)} generations to {output}[/green]")
    if hf_repo:
        console.print(
            f"[green]Pushed to https://huggingface.co/datasets/{hf_repo}[/green]"
        )


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
