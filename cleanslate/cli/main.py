"""CLI entry point and app wiring."""

from __future__ import annotations

import typer
from rich.console import Console

from cleanslate.cli.eval import eval_cmd
from cleanslate.cli.validate import validate_cmd

app = typer.Typer(name="cleanslate", help="Memorization evaluation and unlearning for LLMs.")
console = Console()

app.command(name="eval")(eval_cmd)
app.command(name="validate")(validate_cmd)


def main() -> None:
    app()
