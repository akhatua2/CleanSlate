"""cleanslate score / score-pre / score-post — compute nested-JSON unlearning scores."""

import typer
from rich.console import Console
from rich.table import Table

from cleanslate.fsid.scoring import (
    compute_score,
    compute_score_post,
    compute_score_pre,
    save_score,
)

console = Console()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _print_two_sided_mem(label: str, block: dict) -> None:
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


def _print_two_sided_retention(retention: dict) -> None:
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


def _print_one_sided_mem(label: str, block: dict) -> None:
    table = Table(title=label, show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("n_content", str(block["n_content"]))
    table.add_row("n_windows_total", str(block["n_windows_total"]))
    table.add_row("memorized_windows", str(block["memorized_windows"]))
    table.add_row("mean_memorized_pz", f"{block['mean_memorized_pz']:.6f}")
    console.print(table)


def _print_one_sided_retention(retention: dict) -> None:
    table = Table(title="Retention")
    table.add_column("task", style="bold")
    table.add_column("acc", justify="right")
    table.add_column("n", justify="right")
    for name, row in retention.items():
        table.add_row(name, f"{row['acc']:.4f}", str(row["n"]))
        if name == "cleanslate_qa":
            for sub in ("forget", "retain"):
                if sub in row:
                    s = row[sub]
                    table.add_row(f"  └ {sub}", f"{s['acc']:.4f}", str(s["n"]))
    console.print(table)


def _print_score(score: dict, *, two_sided: bool) -> None:
    mem = _print_two_sided_mem if two_sided else _print_one_sided_mem
    ret = _print_two_sided_retention if two_sided else _print_one_sided_retention
    console.print()
    mem("Forget", score["forget"])
    console.print()
    mem("Retain", score["retain"])
    console.print()
    ret(score["retention"])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def score_cmd(
    forget_pool: str = typer.Option(
        ..., "--forget-pool",
        help="HF id or local path to the forget content_id list (e.g. unlearning-cleanslate/fsid-curated-llama-8b:forget_pool).",
    ),
    pre_eval: str = typer.Option(..., "--pre-eval"),
    post_eval: str = typer.Option(..., "--post-eval"),
    pre_generations: str = typer.Option(..., "--pre-generations"),
    post_generations: str = typer.Option(..., "--post-generations"),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint",
        help="Post-unlearning model spec recorded in meta.checkpoint.",
    ),
    output_path: str = typer.Option("results/fsid/score.json", "--output", "-o"),
) -> None:
    """Two-sided pre→post diff score."""
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
    _print_score(score, two_sided=True)
    console.print(f"\n[bold]Saved to:[/bold] {out}")


def score_pre_cmd(
    forget_pool: str = typer.Option(..., "--forget-pool"),
    pre_eval: str = typer.Option(..., "--pre-eval"),
    pre_generations: str = typer.Option(..., "--pre-generations", "--pre-gen"),
    output_path: str = typer.Option("results/fsid/score_pre.json", "--output", "-o"),
) -> None:
    """Pre-only baseline summary (no post run needed)."""
    console.print("[bold]cleanslate score-pre[/bold] — starting")
    console.print(f"  forget_pool      = {forget_pool}")
    console.print(f"  pre_eval         = {pre_eval}")
    console.print(f"  pre_generations  = {pre_generations}")
    console.print(f"  output_path      = {output_path}")

    score = compute_score_pre(
        forget_pool=forget_pool,
        pre_eval=pre_eval,
        pre_generations=pre_generations,
    )
    out = save_score(score, output_path)
    _print_score(score, two_sided=False)
    console.print(f"\n[bold]Saved to:[/bold] {out}")


def score_post_cmd(
    forget_pool: str = typer.Option(..., "--forget-pool"),
    post_eval: str = typer.Option(..., "--post-eval"),
    post_generations: str = typer.Option(..., "--post-generations", "--post-gen"),
    checkpoint: str | None = typer.Option(
        None, "--checkpoint",
        help="Post-unlearning model spec (HF repo, repo:checkpoint-N, or local path). Recorded in meta.checkpoint.",
    ),
    output_path: str = typer.Option("results/fsid/score_post.json", "--output", "-o"),
) -> None:
    """Post-only summary for spot-checking a freshly-trained checkpoint."""
    console.print("[bold]cleanslate score-post[/bold] — starting")
    console.print(f"  forget_pool      = {forget_pool}")
    console.print(f"  post_eval        = {post_eval}")
    console.print(f"  post_generations = {post_generations}")
    console.print(f"  checkpoint       = {checkpoint}")
    console.print(f"  output_path      = {output_path}")

    score = compute_score_post(
        forget_pool=forget_pool,
        post_eval=post_eval,
        post_generations=post_generations,
        checkpoint=checkpoint,
    )
    out = save_score(score, output_path)
    _print_score(score, two_sided=False)
    console.print(f"\n[bold]Saved to:[/bold] {out}")
