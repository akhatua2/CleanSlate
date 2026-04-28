"""cleanslate score -- compare pre/post memorization artifacts."""

import typer
from rich.console import Console

from cleanslate.fsid.scoring import compute_score, save_score

console = Console()


def score_cmd(
    forget_set_path: str = typer.Option(..., "--forget-path"),
    pre_eval_path: str = typer.Option(..., "--pre-eval"),
    post_eval_path: str = typer.Option(..., "--post-eval"),
    pre_generations_path: str = typer.Option(
        ..., "--pre-generations",
        help="HF dataset ID of pre-unlearning lm-eval generations (e.g. generations-<model>-pre_val).",
    ),
    post_generations_path: str = typer.Option(
        ..., "--post-generations",
        help="HF dataset ID of post-unlearning lm-eval generations.",
    ),
    output_path: str = typer.Option("results/fsid/score.json", "--output", "-o"),
) -> None:
    """Score unlearning by comparing pre/post memorization artifacts."""
    result = compute_score(
        forget_set_path=forget_set_path,
        pre_eval_path=pre_eval_path,
        post_eval_path=post_eval_path,
        pre_generations_path=pre_generations_path,
        post_generations_path=post_generations_path,
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
        console.print(f"  Memorized pre:     {result.forget_windows_memorized_pre}")
        console.print(f"  Memorized post:    {result.forget_windows_memorized_post}")
        console.print(
            f"  Δ memorized:       {result.forget_windows_delta_memorized}  "
            f"(memorized → forgotten)"
        )
        console.print(f"  Forgotten:         {result.forget_windows_forgotten}")
        console.print(f"  Regressed:         {result.forget_windows_regressed}  (clean → memorized)")
        console.print(f"  Forget rate:       {result.forget_window_forget_rate:.1%}")
    console.print()
    console.print("[bold]Retain set[/bold]")
    console.print(f"  Songs scored:      {result.retain_count}")
    console.print(f"  Mean |Δ| frac:     {result.retain_mean_abs_delta_frac:.4f}  (lower = better)")
    console.print(f"  Mean |Δ| max_p_z:  {result.retain_mean_abs_delta_pz:.6f}")
    if result.retain_total_windows is not None:
        console.print(f"  Total windows:     {result.retain_total_windows}")
        console.print(f"  Memorized pre:     {result.retain_windows_memorized_pre}")
        console.print(f"  Memorized post:    {result.retain_windows_memorized_post}")
        console.print(
            f"  Δ memorized:       {result.retain_windows_delta_memorized}  "
            f"(memorized → forgotten on retain content)"
        )
        console.print(f"  Regressed:         {result.retain_windows_regressed}  (clean → memorized)")
        console.print(f"  Stability:         {result.retain_window_stability:.1%}")
    console.print()
    console.print("[bold]Score[/bold]")
    console.print(f"  Forget score:   [green]{result.forget_score:.4f}[/green]")
    console.print(f"  Retain penalty: [yellow]{result.retain_penalty:.4f}[/yellow]")

    if result.generation_deltas:
        console.print()
        console.print("[bold]Validation (per-benchmark, generation-derived)[/bold]")
        for d in result.generation_deltas:
            name = d["benchmark"]
            n_str = f"n={d['n']}"
            if "n_subtasks" in d:
                n_str += f" ({d['n_subtasks']} sub)"
            console.print(
                f"  {name:32s}  pre={d['pre']:.4f}  post={d['post']:.4f}  "
                f"Δ={d['delta']:+.4f}  retention={d['retention']:.4f}  {n_str}"
            )
        if result.bbh_n_subtasks is not None:
            console.print(
                f"  bbh sample-weighted:  pre={result.bbh_pre_sample_weighted:.4f}  "
                f"post={result.bbh_post_sample_weighted:.4f}  Δ={result.bbh_delta_sample_weighted:+.4f}"
            )
        if result.mmlu_n_subtasks is not None:
            console.print(
                f"  mmlu sample-weighted: pre={result.mmlu_pre_sample_weighted:.4f}  "
                f"post={result.mmlu_post_sample_weighted:.4f}  Δ={result.mmlu_delta_sample_weighted:+.4f}"
            )
        if result.utility_retained is not None:
            console.print(f"  Utility retained (equal-weighted):  {result.utility_retained:.1%}")
        if result.utility_retained_sample_weighted is not None:
            console.print(f"  Utility retained (sample-weighted): {result.utility_retained_sample_weighted:.1%}")

    console.print()
    console.print(f"[bold]Saved to:[/bold] {out}")
