"""CLI entry point and app wiring."""

from __future__ import annotations

import typer
from rich.console import Console

from cleanslate.cli.eval import eval_cmd

app = typer.Typer(name="cleanslate", help="Memorization evaluation and unlearning for LLMs.")
console = Console()

app.command(name="eval")(eval_cmd)

# FSID subcommands will be added here:
# from cleanslate.fsid.cli import fsid_app
# app.add_typer(fsid_app, name="fsid")


def main() -> None:
    app()
