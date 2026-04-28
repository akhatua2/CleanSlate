"""cleanslate score -- compare pre/post unlearning artifacts."""

import typer
from rich.console import Console
from rich.table import Table

from cleanslate.fsid.scoring import compute_score, save_score

console = Console()


def _print_mem_block(label: str, block: dict[str, any]) -> None:
    table = Table(title=label, show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("n_content", str(block["n_content"]))
    table.add_row("n_windows_total", str(block["n_windows_total"]))
    table.add_row("pre_memorized_windows", str(block["pre_memorized_windows"]))
    table.add_row("post_memorized_windows", str(block["post_memorized_windows"]))
    table.add_row("delta_memorized_windows", f"{block['delta_memorized_windows']:+d}")
    table.add_row("pre_mean_memorized_pz", f"{block['pre_mean_memorized_pz']:.6f}")
    table.add_row("post_mean_memorized_pz", f"{block['post_mean_memorized_pz']:.6f}")
    console.print(table)


def _print_retention(retention: dict[str, dict[str, any]]) -> None:
    table = Table(title="Retention")
    table.add_column("task", style="bold")
    table.add_column("pre", justify="right")
    table.add_column("post", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("n", justify="right")
    for name, row in retention.items():
        table.add_row(
            name,
            f"{row['pre']:.4f}",
            f"{row['post']:.4f}",
            f"{row['delta']:+.4f}",
            str(row["n"]),
        )
        if name == "cleanslate_qa":
            for sub in ("forget", "retain"):
                if sub in row:
                    s = row[sub]
                    table.add_row(
                        f"  └ {sub}",
                        f"{s['pre']:.4f}",
                        f"{s['post']:.4f}",
                        f"{s['delta']:+.4f}",
                        str(s["n"]),
                    )
    console.print(table)


def score_cmd(
    forget_pool: str = typer.Option(
        ..., "--forget-pool",
        help="HF id or local path to the forget content_id list (e.g. unlearning-cleanslate/fsid-curated-llama-8b:forget_pool).",
    ),
    pre_eval: str = typer.Option(..., "--pre-eval", help="Pre-unlearning memorization eval (HF id or parquet)."),
    post_eval: str = typer.Option(..., "--post-eval", help="Post-unlearning memorization eval (HF id or parquet)."),
    pre_generations: str = typer.Option(
        ..., "--pre-generations",
        help="HF dataset of pre-unlearning lm-eval generations (configs = task names).",
    ),
    post_generations: str = typer.Option(..., "--post-generations"),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint",
        help="Post-unlearning model spec (HF repo, repo:checkpoint-N, or local path) that produced --post-eval / --post-generations. Recorded in meta.checkpoint.",
    ),
    output_path: str = typer.Option("results/fsid/score.json", "--output", "-o"),
) -> None:
    """Compute the nested score JSON from pre/post artifacts."""
    console.print("[bold]cleanslate score[/bold] — starting")
    console.print(f"  forget_pool      = {forget_pool}")
    console.print(f"  pre_eval         = {pre_eval}")
    console.print(f"  post_eval        = {post_eval}")
    console.print(f"  pre_generations  = {pre_generations}")
    console.print(f"  post_generations = {post_generations}")
    console.print(f"  checkpoint       = {checkpoint}")
    console.print(f"  output_path      = {output_path}")

    score = compute_score(
        forget_pool=forget_pool,
        pre_eval=pre_eval,
        post_eval=post_eval,
        pre_generations=pre_generations,
        post_generations=post_generations,
        checkpoint=checkpoint,
    )
    out = save_score(score, output_path)

    console.print()
    _print_mem_block("Forget", score["forget"])
    console.print()
    _print_mem_block("Retain", score["retain"])
    console.print()
    _print_retention(score["retention"])
    console.print()
    console.print(f"[bold]Saved to:[/bold] {out}")
