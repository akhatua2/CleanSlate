"""cleanslate score -- compare pre/post memorization artifacts."""

import typer
from rich.console import Console

from cleanslate.fsid.scoring import compute_score, save_score

console = Console()


def score_cmd(
    forget_set_path: str = typer.Option(..., "--forget-path"),
    pre_eval_path: str = typer.Option(..., "--pre-eval"),
    post_eval_path: str = typer.Option(..., "--post-eval"),
    pre_validation_path: str | None = typer.Option(None, "--pre-validation"),
    post_validation_path: str | None = typer.Option(None, "--post-validation"),
    output_path: str = typer.Option("results/fsid/score.json", "--output", "-o"),
) -> None:
    """Score unlearning by comparing pre/post memorization artifacts."""
    result = compute_score(
        forget_set_path=forget_set_path,
        pre_eval_path=pre_eval_path,
        post_eval_path=post_eval_path,
    )
    out = save_score(result, output_path)

    console.print()
    console.print("[bold]Forget set[/bold]")
    console.print(f"  Songs scored:      {result.forget_count}")
    console.print(f"  Fraction improved: {result.forget_fraction_improved:.1%}")
    console.print(
        f"  Mean Δ frac:       {result.forget_mean_delta_frac:+.4f}  (positive = less memorized)"
    )
    console.print(f"  Mean Δ max_p_z:    {result.forget_mean_delta_pz:+.6f}")
    if result.forget_total_windows is not None:
        console.print(f"  Total windows:     {result.forget_total_windows}")
        console.print(f"  Forgotten:         {result.forget_windows_forgotten}")
        console.print(f"  Regressed:         {result.forget_windows_regressed}")
        console.print(f"  Forget rate:       {result.forget_window_forget_rate:.1%}")
    console.print()
    console.print("[bold]Retain set[/bold]")
    console.print(f"  Songs scored:      {result.retain_count}")
    console.print(f"  Mean |Δ| frac:     {result.retain_mean_abs_delta_frac:.4f}  (lower = better)")
    console.print(f"  Mean |Δ| max_p_z:  {result.retain_mean_abs_delta_pz:.6f}")
    if result.retain_total_windows is not None:
        console.print(f"  Total windows:     {result.retain_total_windows}")
        console.print(f"  Regressed:         {result.retain_windows_regressed}")
        console.print(f"  Stability:         {result.retain_window_stability:.1%}")
    console.print()
    console.print("[bold]Score[/bold]")
    console.print(f"  Forget score:   [green]{result.forget_score:.4f}[/green]")
    console.print(f"  Retain penalty: [yellow]{result.retain_penalty:.4f}[/yellow]")

    if pre_validation_path and post_validation_path:
        from cleanslate.validation import compare_validations, load_validation

        pre_val = load_validation(pre_validation_path)
        post_val = load_validation(post_validation_path)
        deltas = compare_validations(pre_val, post_val)
        if deltas:
            console.print()
            console.print("[bold]Validation[/bold]")
            for d in deltas:
                console.print(
                    f"  {d.benchmark_name.upper():10s}  pre={d.pre_accuracy:.1%}  "
                    f"post={d.post_accuracy:.1%}  Δ={d.delta_accuracy:+.1%}  "
                    f"({d.relative_retention:.1%} retained)"
                )
            avg_retention = sum(d.relative_retention for d in deltas) / len(deltas)
            console.print(f"  Utility retained: {avg_retention:.1%}")

    console.print()
    console.print(f"[bold]Saved to:[/bold] {out}")
