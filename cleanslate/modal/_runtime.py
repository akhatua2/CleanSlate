"""Shared Modal runtime: image, volumes, secrets.

These are deliberately not imported at module-level outside this package so
that ``import cleanslate`` remains usable on hosts without ``modal`` installed.
"""

from __future__ import annotations

import modal


def build_training_image() -> modal.Image:
    """Image used by training jobs.

    Bakes the local ``cleanslate`` source in via ``add_local_python_source`` so
    in-progress changes (e.g. the ``io.py`` HF-path patch) are picked up
    without a published release. Vllm is intentionally absent — training does
    not need it and excluding it keeps the image ~3 GB lighter.
    """
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install("git")
        .uv_pip_install(
            "torch==2.10.0",
            "transformers==5.6.0",
            "datasets==4.7.0",
            "accelerate>=0.34",
            "bitsandbytes>=0.49.2",
            "pyarrow==23.0.1",
            "pyyaml==6.0.3",
            "typer==0.24.1",
            "rich==14.3.3",
            "tqdm==4.67.3",
            "hf-transfer==0.1.9",
            # `packaging` and `setuptools` are explicit because mamba-ssm's setup.py
            # imports them; with --no-build-isolation they must come from the env.
            "packaging",
            "setuptools",
            "wheel",
            "ninja",
        )
        # mamba-ssm is required by Nemotron-Nano-9B-v2's hybrid Mamba+Transformer
        # architecture; transformers raises `ImportError: mamba-ssm is required`
        # otherwise. Installed with `--no-build-isolation` because the package
        # has CUDA extensions that import torch at build time.
        #
        # `causal-conv1d` is intentionally NOT installed — its from-source build
        # consistently fails on Blackwell (B200, compute cap 10.0) and mamba-ssm
        # has a slow-path fallback when it's missing. If the slow path is too
        # slow, swap in a prebuilt causal-conv1d wheel from the upstream
        # release page (Dao-AILab/causal-conv1d).
        .pip_install(
            "mamba-ssm>=2.2.0",
            extra_options="--no-build-isolation",
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/model_cache",
            }
        )
        .add_local_python_source("cleanslate")
    )


def build_eval_image() -> modal.Image:
    """Image used by memorization eval jobs.

    Includes vLLM (which the training image excludes) because
    ``MemorizationEvaluator`` runs vLLM in-process for prompt-logprob
    extraction.
    """
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .apt_install("git")
        .uv_pip_install(
            "vllm==0.19.1",
            "torch==2.10.0",
            "transformers==5.6.0",
            "datasets==4.7.0",
            "pyarrow==23.0.1",
            "pyyaml==6.0.3",
            "typer==0.24.1",
            "rich==14.3.3",
            "tqdm==4.67.3",
            "hf-transfer==0.1.9",
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_HOME": "/model_cache",
            }
        )
        .add_local_python_source("cleanslate")
    )


HF_CACHE = modal.Volume.from_name("hf-cache", create_if_missing=True)
RUNS_VOLUME = modal.Volume.from_name("training-runs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")
