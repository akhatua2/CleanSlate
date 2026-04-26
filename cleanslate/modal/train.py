"""Run a single FSID training job on Modal.

Invoked from the local CLI via ``cleanslate fsid train ... --modal``.
Constructs an ephemeral Modal app per call (App + ``@app.function`` are built
inside ``run_remote`` so the GPU spec is captured at decoration time, working
around Modal 1.4.x not exposing ``Function.with_options``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cleanslate.fsid.config import TrainerConfig

logger = logging.getLogger(__name__)


def _app_name(config: TrainerConfig) -> str:
    """Modal app name: ``<model_clean>-train``.

    Intentionally generic — no algorithm, dataset, or project name leaks into
    the Modal dashboard. The ``-train`` suffix separates training apps from
    the matching ``modal_vllm_server.py`` ``<model_clean>`` serving apps.
    """
    model_clean = (
        config.model_name.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    )
    return f"{model_clean}-train"


def run_remote(
    config: TrainerConfig,
    *,
    gpu: str = "H100",
    timeout_seconds: int = 12 * 3600,
) -> dict[str, Any]:
    """Run ``train_unlearning(config)`` in an ephemeral Modal container.

    ``gpu`` is a Modal GPU spec (``"H100"``, ``"H100:8"``, ``"A100-80GB:4"``).
    For ``GPU:N`` with ``N>1`` the inner function launches via ``torchrun`` so
    the FSDP path is identical to local multi-GPU runs.

    Output is forced to ``/runs`` (the persistent volume) regardless of the
    caller's ``output_dir`` so checkpoints survive container teardown. Pull
    them later with ``modal volume get training-runs <run_name> ./local/`` or
    set ``push_to_hub=True`` in the config.
    """
    import modal

    from cleanslate.modal._runtime import (
        HF_CACHE,
        HF_SECRET,
        RUNS_VOLUME,
        build_training_image,
    )

    config_dict = asdict(config)
    config_dict["output_dir"] = "/runs"

    gpu_type, _, gpu_count_str = gpu.partition(":")
    gpu_count = int(gpu_count_str) if gpu_count_str else 1

    image = build_training_image()
    app = modal.App(_app_name(config), image=image)

    @app.function(
        gpu=gpu,
        secrets=[HF_SECRET],
        volumes={"/model_cache": HF_CACHE, "/runs": RUNS_VOLUME},
        timeout=timeout_seconds,
        serialized=True,
    )
    def remote_train(payload: dict[str, Any]) -> dict[str, Any]:
        from pathlib import Path

        cfg = payload["config"]
        gpus = int(payload["gpu_count"])

        if gpus > 1:
            _run_torchrun(cfg, gpus)
        else:
            _run_inproc(cfg)

        return _resolve_summary(Path("/runs"))

    payload = {"config": config_dict, "gpu_count": gpu_count}
    with modal.enable_output():
        with app.run():
            return remote_train.remote(payload)


def _run_inproc(cfg: dict[str, Any]) -> None:
    """Single-GPU path: import and call ``train_unlearning`` directly.

    Avoids spawning a subprocess so HF Trainer logs stream cleanly through
    Modal's stdout pipe.
    """
    from cleanslate.fsid.config import TrainerConfig
    from cleanslate.fsid.training.trainer import train_unlearning

    config = TrainerConfig(**cfg)
    train_unlearning(config)


def _run_torchrun(cfg: dict[str, Any], gpus: int) -> None:
    """Multi-GPU path: torchrun an inline launcher that calls train_unlearning.

    Sidesteps the ``cleanslate`` console-script dependency (the package is
    surfaced via ``add_local_python_source``, which makes it importable but
    does not register console entrypoints).
    """
    import subprocess
    import sys
    import tempfile

    launcher = (
        "import json, sys\n"
        "from cleanslate.fsid.config import TrainerConfig\n"
        "from cleanslate.fsid.training.trainer import train_unlearning\n"
        "with open(sys.argv[1]) as f:\n"
        "    cfg_dict = json.load(f)\n"
        "config = TrainerConfig(**cfg_dict)\n"
        "train_unlearning(config)\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as launcher_f:
        launcher_f.write(launcher)
        launcher_path = launcher_f.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as cfg_f:
        json.dump(cfg, cfg_f)
        cfg_path = cfg_f.name

    subprocess.run(
        [
            "torchrun",
            f"--nproc_per_node={gpus}",
            launcher_path,
            cfg_path,
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _resolve_summary(runs_root: Any) -> dict[str, Any]:
    """Find the most recently created run dir and return its train_summary.json."""
    runs = sorted(runs_root.glob("*/"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f"Training completed but {runs_root} is empty")
    latest = runs[-1]
    summary_path = latest / "train_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"No train_summary.json in {latest}")
    return json.loads(summary_path.read_text())
