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
from .results import ExperimentResults
from .samples import SHAKESPEARE_SAMPLES, BOOK_SAMPLES, SONG_SAMPLES

app = typer.Typer(
    name="cleanslate",
    help="Sequence likelihood estimation for LLMs - measure memorization and token probabilities.",
)
console = Console()


def _print_inline_diff(console: Console, result) -> None:
    """Print inline diff for a sample result."""
    if not result.target_tokens or not result.predicted_tokens:
        return

    # Build inline diff view
    target_line = ""
    predicted_line = ""

    for tgt, pred, rank in zip(
        result.target_tokens, result.predicted_tokens, result.token_ranks
    ):
        if rank == 1:
            target_line += f"[green]{tgt}[/green]"
            predicted_line += f"[green]{pred}[/green]"
        else:
            target_line += f"[red]{tgt}[/red]"
            predicted_line += f"[yellow]{pred}[/yellow]"

    console.print(f"[dim]Target:[/dim]    {target_line}")
    console.print(f"[dim]Predicted:[/dim] {predicted_line}")


@app.command()
def run(
    model: str = typer.Option("Qwen/Qwen3-0.6B", "--model", "-m", help="Model name"),
    dataset: str = typer.Option(
        "shakespeare", "--dataset", "-d", help="Dataset: shakespeare, books, songs, or all"
    ),
    name: str = typer.Option("experiment", "--name", "-n", help="Experiment name"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON file path"
    ),
    device: Optional[str] = typer.Option(
        None, "--device", help="Device: cpu, cuda, mps, or auto"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show token-by-token diff for each sample"
    ),
):
    """Run sequence likelihood experiment on a dataset."""

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

    # Show verbose diff for each sample
    if verbose:
        console.print()
        console.print("[bold]Token Diffs per Sample:[/bold]")
        for sample in results.samples:
            console.print()
            console.print(f"[bold cyan]{sample.name}[/bold cyan]")
            _print_inline_diff(console, sample)

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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show token-by-token diff"
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

    # Show token diff if verbose
    if verbose:
        console.print()
        console.print("[bold]Inline Diff:[/bold]")
        console.print()
        _print_inline_diff(console, result)


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

    console.print()
    console.print("[bold]Song Samples:[/bold]")
    for i, s in enumerate(SONG_SAMPLES, 1):
        console.print(f"  {i}. {s['name']}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
