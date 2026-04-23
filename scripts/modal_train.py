"""FSID unlearning training on Modal.

Three GPU-tier functions wrap a shared training routine. Each invocation runs
``cleanslate fsid train`` (or ``accelerate launch`` for multi-GPU) inside a Modal
container, then uploads the resulting checkpoint to HF Hub under the convention
``unlearning-cleanslate/<hf_basename>-<preset>-<rule>``.

Usage (single combo, debug):
    uv run modal run scripts/modal_train.py::train_1gpu \\
        --model qwen-8b --rule baseline --objective simnpo --preset simnpo

Sweep entrypoint lives in ``scripts/sweep_train.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

REPO = "https://github.com/akhatua2/CleanSlate.git"
BRANCH = os.environ.get("CLEANSLATE_BRANCH", "main")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .run_commands(f"git clone -b {BRANCH} {REPO} /app")
    .uv_pip_install("-e", "/app[unlearning]", "hf-transfer==0.1.9")
    .env({
        "HF_HOME": "/model_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .workdir("/app")
)

app = modal.App("fsid-train")

model_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("fsid-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

VOLUMES = {"/model_cache": model_cache, "/runs": runs_vol}


# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------

# Convention: `unlearning-cleanslate/fsid_<alias>:forget_<rule>` / `:retain_<rule>`.
# Curation pipelines (any source) publish parquets to this path; training pulls them.
def forget_path(model_alias: str, rule: str) -> str:
    return f"unlearning-cleanslate/fsid_{model_alias}:forget_{rule}"


def retain_path(model_alias: str, rule: str) -> str:
    return f"unlearning-cleanslate/fsid_{model_alias}:retain_{rule}"


def hub_repo_name(hf_model: str, preset: str, rule: str) -> str:
    """Match the existing convention: ``olmo-3-1125-32b-simnpo-gentle-corpus-sweep``."""
    base = hf_model.split("/")[-1].lower()
    return f"{base}-{preset.replace('_', '-')}-{rule.replace('_', '-')}"


# ---------------------------------------------------------------------------
# Shared training implementation
# ---------------------------------------------------------------------------

def _resolve_model(alias: str) -> str:
    """Resolve a `cleanslate` alias to its HF path. Mirror of cli/aliases.py."""
    import sys
    sys.path.insert(0, "/app")
    from cleanslate.cli.aliases import resolve_model
    return resolve_model(alias)[0]


def _train_impl(
    model_alias: str,
    rule: str,
    objective: str,
    preset: str,
    n_gpus: int,
    forget_override: str | None = None,
    retain_override: str | None = None,
) -> str:
    """Run training + upload. Returns repo URL on success, raises otherwise.

    forget_override / retain_override let callers point at non-convention Hub paths
    (or local files) when curated parquets live somewhere other than
    fsid_<model>:forget_<rule> / :retain_<rule>.
    """
    from huggingface_hub import HfApi  # type: ignore[import-untyped]

    hf_model = _resolve_model(model_alias)
    repo = f"unlearning-cleanslate/{hub_repo_name(hf_model, preset, rule)}"

    api = HfApi()
    if api.repo_exists(repo, repo_type="model"):
        return f"SKIP (exists): {repo}"

    config_path = f"/app/cleanslate/fsid/configs/{preset}.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"preset config not found: {config_path}")

    output_dir = f"/runs/{model_alias}/{rule}/{preset}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cli_args = [
        "fsid", "train",
        "--config", config_path,
        "-m", model_alias,
        "--objective", objective,
        "--forget-path", forget_override or forget_path(model_alias, rule),
        "--retain-path", retain_override or retain_path(model_alias, rule),
        "--output-dir", output_dir,
    ]
    if n_gpus > 1:
        cli_args += ["--distributed-strategy", "fsdp"]
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(n_gpus),
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "-m", "cleanslate.cli", *cli_args,
        ]
    else:
        cmd = ["cleanslate", *cli_args]

    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/app")
    if proc.returncode != 0:
        raise RuntimeError(f"training failed (exit={proc.returncode}) for {model_alias}/{rule}/{preset}")

    # `train_unlearning` writes to <output_dir>/<run_name>/. Pick newest.
    run_dirs = sorted(Path(output_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    run_dirs = [d for d in run_dirs if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"no run directory under {output_dir}")
    ckpt_dir = run_dirs[0]

    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    upload = subprocess.run(
        ["huggingface-cli", "upload", repo, str(ckpt_dir), ".", "--repo-type", "model"],
        cwd="/app",
    )
    if upload.returncode != 0:
        raise RuntimeError(f"hub upload failed for {repo}")

    runs_vol.commit()
    return f"OK: https://huggingface.co/{repo}"


# ---------------------------------------------------------------------------
# GPU-tier functions
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=4 * 60 * 60,
)
def train_1gpu(
    model: str, rule: str, objective: str, preset: str,
    forget_override: str | None = None, retain_override: str | None = None,
) -> str:
    return _train_impl(model, rule, objective, preset, n_gpus=1,
                       forget_override=forget_override, retain_override=retain_override)


@app.function(
    image=train_image,
    gpu="H100:2",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=6 * 60 * 60,
)
def train_2gpu(
    model: str, rule: str, objective: str, preset: str,
    forget_override: str | None = None, retain_override: str | None = None,
) -> str:
    return _train_impl(model, rule, objective, preset, n_gpus=2,
                       forget_override=forget_override, retain_override=retain_override)


@app.function(
    image=train_image,
    gpu="H100:8",
    volumes=VOLUMES,
    secrets=[hf_secret],
    timeout=12 * 60 * 60,
)
def train_8gpu(
    model: str, rule: str, objective: str, preset: str,
    forget_override: str | None = None, retain_override: str | None = None,
) -> str:
    return _train_impl(model, rule, objective, preset, n_gpus=8,
                       forget_override=forget_override, retain_override=retain_override)
