"""Parallel memorization eval across models using OpenModal.

Usage:
    cd ~/openmodal && uv run openmodal run ~/CleanSlate/scripts/parallel_eval.py
"""

import openmodal

app = openmodal.App("cleanslate-eval")

REPO = "https://github.com/akhatua2/CleanSlate.git"
BRANCH = "new_v2"

eval_image = (
    openmodal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "vllm==0.19.0",
        "torch==2.10.0",
        "transformers==4.57.6",
        "pyarrow==23.0.1",
        "pyyaml==6.0.3",
        "datasets==4.7.0",
        "typer==0.24.1",
        "rich==14.3.3",
        "tqdm==4.67.3",
        "hf-transfer==0.1.9",
    )
    .run_commands(
        f"git clone -b {BRANCH} {REPO} /app",
        "cd /app && pip install -e .",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .workdir("/app")
)

model_cache = openmodal.Volume.from_name("cleanslate-model-cache", create_if_missing=True)
results_vol = openmodal.Volume.from_name("cleanslate-results", create_if_missing=True)
hf_secret = openmodal.Secret.from_name("huggingface-secret")

# Skip models already running locally: llama-8b, qwen-8b, olmo-7b, gemma-12b, marin-8b
MODELS_1GPU = [
    "qwen2-7b", "qwen2.5-7b", "qwen3.5-9b",
    "qwen2.5-7b-instruct", "qwen2.5-coder-7b", "qwen2.5-coder-7b-instruct",
    "qwen2.5-math-7b", "qwen2.5-math-7b-instruct",
]

MODELS_2GPU = [
    "olmo-32b", "qwen-32b", "gemma-27b", "marin-32b",
]


@app.function(
    image=eval_image,
    gpu="H100",
    volumes={"/model_cache": model_cache, "/results": results_vol},
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def eval_1gpu(model_alias: str) -> str:
    import subprocess
    r = subprocess.run(
        ["cleanslate", "eval", "-m", model_alias, "-o", "/results/eval", "--no-push"],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return f"FAIL: {model_alias}"
    return f"OK: {model_alias}"


@app.function(
    image=eval_image,
    gpu="H100:2",
    volumes={"/model_cache": model_cache, "/results": results_vol},
    secrets=[hf_secret],
    timeout=8 * 60 * 60,
)
def eval_2gpu(model_alias: str) -> str:
    import subprocess
    r = subprocess.run(
        ["cleanslate", "eval", "-m", model_alias, "-o", "/results/eval",
         "--no-push", "--config", "config_32b.yaml"],
        capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return f"FAIL: {model_alias}"
    return f"OK: {model_alias}"


@app.local_entrypoint()
def main():
    print(f"Launching {len(MODELS_1GPU)} x 1-GPU + {len(MODELS_2GPU)} x 2-GPU evals")

    results = []
    for r in eval_1gpu.map(MODELS_1GPU):
        print(r)
        results.append(r)

    for r in eval_2gpu.map(MODELS_2GPU):
        print(r)
        results.append(r)

    ok = sum(1 for r in results if r.startswith("OK"))
    print(f"\nDone: {ok}/{len(results)} succeeded")
