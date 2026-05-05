"""Push training checkpoint dirs from Modal training-runs volume to HF Hub."""
import modal

RUNS_VOLUME = modal.Volume.from_name("training-runs")
HF_SECRET = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("push-ckpts-to-hf", image=image)


@app.function(
    secrets=[HF_SECRET],
    volumes={"/runs": RUNS_VOLUME},
    timeout=3 * 3600,
    cpu=4,
)
def push_run(run_path: str, hf_repo_id: str) -> str:
    from huggingface_hub import HfApi
    import os

    api = HfApi()
    local_path = f"/runs/{run_path}"
    print(f"HF_TOKEN present: {'HF_TOKEN' in os.environ}")
    print(f"Pushing {local_path} -> {hf_repo_id}")
    api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="upload checkpoints from Modal training-runs volume",
    )
    msg = f"Done: https://huggingface.co/{hf_repo_id}"
    print(msg)
    return msg


@app.local_entrypoint()
def main() -> None:
    # Recover olmo-32b cell42 (bm25_6T) from disk-full crash:
    # trainer.save_model() to /runs root failed mid-write, but the
    # intermediate checkpoint-11 saved cleanly with all 3 model shards +
    # tokenizer + config. Push that directly.
    jobs = [
        ("simnpo_Olmo-3-1125-32B_20260504_000229/checkpoint-11",
         "arpandeepk/olmo-3-32b-simnpo-gentle-bm25-6t"),
    ]
    # Push in parallel — different repos, no contention
    handles = [
        (run_path, hf_repo_id, push_run.spawn(run_path, hf_repo_id))
        for run_path, hf_repo_id in jobs
    ]
    for run_path, hf_repo_id, h in handles:
        print(f"\n--- Waiting on {run_path} -> {hf_repo_id} ---")
        print(h.get())
