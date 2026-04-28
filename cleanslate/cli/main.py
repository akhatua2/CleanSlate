"""CLI entry point and app wiring."""

from __future__ import annotations

import atexit
import os

import typer
from rich.console import Console

from cleanslate.cli.eval import eval_cmd
from cleanslate.cli.fsid import fsid_app
from cleanslate.cli.score import score_cmd, score_post_cmd, score_pre_cmd
from cleanslate.cli.validate import generate_cmd, judge_cmd, validate_cmd

app = typer.Typer(name="cleanslate", help="Memorization evaluation and unlearning for LLMs.")
console = Console()

app.command(name="eval")(eval_cmd)
app.command(name="generate")(generate_cmd)
app.command(name="judge")(judge_cmd)
app.command(name="validate")(validate_cmd)
app.command(name="score")(score_cmd)
app.command(name="score-pre")(score_pre_cmd)
app.command(name="score-post")(score_post_cmd)
app.add_typer(fsid_app)


def _silence_stderr_at_exit() -> None:
    """Suppress vLLM child-process ResourceTracker noise on shutdown."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        pass


def main() -> None:
    atexit.register(_silence_stderr_at_exit)
    app()
