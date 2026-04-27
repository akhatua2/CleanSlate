"""cleanslate eval -- sliding window memorization evaluation."""

from __future__ import annotations

import typer

from cleanslate.eval_core import DEFAULT_DATASET, run_memorization_eval


def eval_cmd(
    hf_dataset: str = typer.Option(DEFAULT_DATASET, "-d", "--hf-dataset"),
    hf_config: str = typer.Option("default", "-c", "--hf-config"),
    text_col: str = typer.Option("reference_target", "--text-col"),
    limit: int | None = typer.Option(None, "-n", "--limit"),
    offset: int = typer.Option(0, "--offset"),
    model: str = typer.Option("Qwen/Qwen3-8B", "-m", "--model"),
    output_dir: str = typer.Option("results/eval", "-o", "--output"),
    hf_org: str = typer.Option("unlearning-cleanslate", "--hf-org"),
    no_push: bool = typer.Option(False, "--no-push"),
    push_suffix: str | None = typer.Option(
        None, "--push-suffix",
        help="Suffix appended to the HF repo name: eval-<model_clean>-<suffix>.",
    ),
    config_path: str | None = typer.Option(None, "--config", help="Path to YAML config file"),
    on_modal: bool = typer.Option(
        False, "--modal",
        help="Run remotely on Modal instead of locally. Pushes results to HF and to the 'training-runs' Modal volume.",
    ),
    modal_gpu: str = typer.Option(
        "H100", "--modal-gpu",
        help="Modal GPU spec, e.g. 'H100', 'B200', 'A100-80GB'. Single-GPU only.",
    ),
) -> None:
    """Sliding window memorization evaluation.

    For each text: slide a character-based window, compute p_z via teacher forcing.
    """
    params = {
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "text_col": text_col,
        "limit": limit,
        "offset": offset,
        "model": model,
        "output_dir": output_dir,
        "hf_org": hf_org,
        "no_push": no_push,
        "push_suffix": push_suffix,
        "config_path": config_path,
    }
    if on_modal:
        from cleanslate.modal.eval import run_remote

        run_remote(params, gpu=modal_gpu)
        return
    run_memorization_eval(params)
