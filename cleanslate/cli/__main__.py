"""Enable `python -m cleanslate.cli ...` (used by `accelerate launch -m cleanslate.cli`)."""

from cleanslate.cli.main import main

if __name__ == "__main__":
    main()
