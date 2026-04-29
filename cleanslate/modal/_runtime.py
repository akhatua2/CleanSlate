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
        # mamba-ssm + causal-conv1d are required by Nemotron-Nano-9B-v2's hybrid
        # Mamba+Transformer architecture. Without causal-conv1d's `causal_conv1d_fn`,
        # mamba falls back to a naive PyTorch path that is ~30× slower.
        #
        # By default causal-conv1d's setup.py tries to download a prebuilt wheel;
        # for our torch 2.10 + CUDA 12.8 + py3.12 combination there is no wheel
        # → HTTP 404 → error handler crashes with TypeError. CAUSAL_CONV1D_FORCE_BUILD
        # / MAMBA_FORCE_BUILD bypass the wheel lookup and go straight to source build.
        #
        # The source build needs TORCH_CUDA_ARCH_LIST set so it builds for the
        # right arches. We set "9.0+PTX" — the Hopper kernel + PTX bytecode that
        # forward-JITs to sm_100 (Blackwell) at first kernel launch.
        .env({
            "TORCH_CUDA_ARCH_LIST": "9.0+PTX",
            "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
            "MAMBA_FORCE_BUILD": "TRUE",
        })
        .pip_install(
            "causal-conv1d>=1.4.0",
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
