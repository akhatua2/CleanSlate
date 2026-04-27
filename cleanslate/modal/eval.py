"""Run a memorization eval on Modal.

Invoked from the local CLI via ``cleanslate eval ... --modal``. Builds an
ephemeral Modal app per call and runs ``MemorizationEvaluator`` inside a single
GPU container. Results land in HF (the existing default of ``cleanslate eval``)
and on the persistent ``training-runs`` volume under ``/runs/eval/<model_clean>/``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _app_name(model: str) -> str:
    """Modal app name: ``<model_clean>-eval``.

    Generic — no algorithm, dataset, or project name leaks into the Modal
    dashboard. The ``-eval`` suffix separates eval apps from training (``-train``)
    and serving (``<model_clean>``) apps.
    """
    model_clean = model.split("/")[-1].lower().replace(".", "-").replace("_", "-")
    return f"{model_clean}-eval"


def run_remote(
    params: dict[str, Any],
    *,
    gpu: str = "H100",
    timeout_seconds: int = 60 * 60,
) -> None:
    """Run ``run_memorization_eval(params)`` in an ephemeral Modal container.

    ``gpu`` is a Modal GPU spec (``"H100"``, ``"B200"``, ``"A100-80GB"``).
    Memorization eval is single-GPU only — vLLM handles internal parallelism.

    ``output_dir`` in the caller's params is rewritten to ``/runs/eval/<model_clean>``
    (on the persistent volume) so the Parquet+JSON survive container teardown.
    HF push (default on) is unaffected.
    """
    import modal

    from cleanslate.modal._runtime import (
        HF_CACHE,
        HF_SECRET,
        RUNS_VOLUME,
        build_eval_image,
    )

    model = params["model"]
    model_clean = model.split("/")[-1].lower().replace(".", "_")
    payload = dict(params)
    payload["output_dir"] = f"/runs/eval/{model_clean}"

    image = build_eval_image()
    app = modal.App(_app_name(model), image=image)

    @app.function(
        gpu=gpu,
        secrets=[HF_SECRET],
        volumes={"/model_cache": HF_CACHE, "/runs": RUNS_VOLUME},
        timeout=timeout_seconds,
        serialized=True,
    )
    def remote_eval(p: dict[str, Any]) -> None:
        from cleanslate.eval_core import run_memorization_eval

        run_memorization_eval(p)

    with modal.enable_output():
        with app.run():
            remote_eval.remote(payload)
