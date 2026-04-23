"""Entrypoint so `python -m cleanslate` works (needed for `accelerate launch -m cleanslate`)."""

from cleanslate.cli import main

if __name__ == "__main__":
    main()
