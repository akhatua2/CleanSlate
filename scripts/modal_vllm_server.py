"""Serve a vLLM OpenAI-compatible endpoint on Modal.

Deploy one app per model (app name is derived from MODEL).

Usage:
    MODEL=meta-llama/Llama-3.1-8B uv run modal deploy scripts/modal_vllm_server.py
    MODEL=Qwen/Qwen3-8B            uv run modal deploy scripts/modal_vllm_server.py
    MODEL=allenai/Olmo-3-1025-7B   uv run modal deploy scripts/modal_vllm_server.py
    MODEL=allenai/Olmo-3-1125-32B GPU_COUNT=2 uv run modal deploy scripts/modal_vllm_server.py

After deploy, `modal app list` shows the URL, or call:
    modal run scripts/modal_vllm_server.py::url
"""

from __future__ import annotations

import os

import modal

MODEL = os.environ["MODEL"]
GPU_TYPE = os.environ["GPU_TYPE"]
GPU_COUNT = int(os.environ["GPU_COUNT"])
MAX_MODEL_LEN = int(os.environ["MAX_MODEL_LEN"])
VLLM_PORT = 8000

APP_NAME = os.environ.get("APP_NAME", MODEL.split("/")[-1].lower().replace(".", "-").replace("_", "-"))
# When serving a checkpoint from a volume path, set this to the base HF model
# name so clients (lm-eval) can look up the tokenizer and the served-model-name
# matches what they send in requests.
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "vllm==0.19.1",
        "torch==2.10.0",
        "transformers==5.6.0",
        "hf-transfer==0.1.9",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/model_cache",
        "MODEL": MODEL,
        "GPU_TYPE": GPU_TYPE,
        "GPU_COUNT": str(GPU_COUNT),
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
    })
)

app = modal.App(APP_NAME)

hf_secret = modal.Secret.from_name("huggingface")
model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("training-runs", create_if_missing=False)

gpu_spec = f"{GPU_TYPE}:{GPU_COUNT}" if GPU_COUNT > 1 else GPU_TYPE


@app.function(
    image=vllm_image,
    gpu=gpu_spec,
    secrets=[hf_secret],
    volumes={"/model_cache": model_cache, "/runs": runs_volume},
    timeout=60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * 60)
def serve() -> None:
    import subprocess

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", SERVED_MODEL_NAME,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    if GPU_COUNT > 1:
        cmd.extend(["--tensor-parallel-size", str(GPU_COUNT)])

    subprocess.Popen(cmd)


@app.local_entrypoint()
def url() -> None:
    print(f"App:  {APP_NAME}")
    print(f"URL:  {serve.get_web_url()}")
