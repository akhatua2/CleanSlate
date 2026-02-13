"""Run cleanslate eval in parallel across multiple GPUs."""

import argparse
import logging
import os
import subprocess
import threading
import queue
from dataclasses import dataclass

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvalJob:
    """A single eval job for one GPU."""

    gpu_id: str
    part_id: str
    offset: int
    limit: int


def get_dataset_size(hf_dataset: str, hf_config: str) -> int:
    """Get total number of rows in the dataset."""
    logger.info(f"Checking dataset size: {hf_dataset} / {hf_config}")
    ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)
    # Count rows (streaming so we iterate)
    count = 0
    for _ in ds:
        count += 1
    logger.info(f"Dataset has {count} rows")
    return count


def create_jobs(gpus: list[str], total_rows: int, limit: int | None) -> list[EvalJob]:
    """Create jobs for each GPU with offset/limit splits."""
    # If user specified a limit, use that as total
    effective_total = min(total_rows, limit) if limit else total_rows

    num_gpus = len(gpus)
    rows_per_gpu = effective_total // num_gpus
    remainder = effective_total % num_gpus

    jobs = []
    current_offset = 0

    for i, gpu_id in enumerate(gpus):
        # Distribute remainder across first few GPUs
        job_limit = rows_per_gpu + (1 if i < remainder else 0)

        if job_limit > 0:
            jobs.append(
                EvalJob(
                    gpu_id=gpu_id,
                    part_id=f"part_{i}",
                    offset=current_offset,
                    limit=job_limit,
                )
            )
            current_offset += job_limit

    return jobs


def run_eval_job(job: EvalJob, base_args: list[str]) -> int:
    """Run a single eval job on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu_id

    # Move ALL caches to DFS to avoid AFS quota/permission issues
    dfs_cache = "/dfs/project/kgrlm/akhatua/.cache"
    env["FLASHINFER_CACHE_DIR"] = f"{dfs_cache}/flashinfer"
    env["HF_HOME"] = f"{dfs_cache}/huggingface"
    env["TORCH_HOME"] = f"{dfs_cache}/torch"
    env["TRITON_CACHE_DIR"] = f"{dfs_cache}/triton"
    env["XDG_CACHE_HOME"] = dfs_cache
    env["HOME"] = "/dfs/project/kgrlm/akhatua"  # Override HOME to avoid any ~/.* paths

    cmd = [
        "cleanslate",
        "eval",
        *base_args,
        "--offset",
        str(job.offset),
        "--limit",
        str(job.limit),
        "--part-id",
        job.part_id,
    ]

    logger.info(f"[GPU {job.gpu_id}] Starting {job.part_id}: offset={job.offset}, limit={job.limit}")
    logger.debug(f"[GPU {job.gpu_id}] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        logger.error(f"[GPU {job.gpu_id}] {job.part_id} failed with exit code {result.returncode}")
    else:
        logger.info(f"[GPU {job.gpu_id}] {job.part_id} completed successfully")

    return result.returncode


def worker(job_queue: "queue.Queue[EvalJob]", base_args: list[str], results: dict[str, int]) -> None:
    """Worker thread that processes jobs from the queue."""
    while True:
        try:
            job = job_queue.get_nowait()
        except queue.Empty:
            return

        try:
            returncode = run_eval_job(job, base_args)
            results[job.part_id] = returncode
        except Exception as e:
            logger.exception(f"[GPU {job.gpu_id}] Exception running {job.part_id}: {e}")
            results[job.part_id] = -1
        finally:
            job_queue.task_done()


def run_merge(input_dir: str, hf_config: str, hf_push: str | None) -> int:
    """Run the merge script."""
    cmd = [
        "python",
        "scripts/merge_eval_parts.py",
        "--input-dir",
        input_dir,
        "--hf-config",
        hf_config,
    ]
    if hf_push:
        cmd.extend(["--hf-push", hf_push])

    logger.info("Running merge script...")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cleanslate eval in parallel across multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/eval_parallel.py \\
    --gpus 1,2,3,4,5,6 \\
    -d unlearning-cleanslate/generated_songs \\
    -c direct__first_line \\
    -m "Qwen/Qwen3-8B" \\
    --window-size 50 \\
    --generate-all \\
    --hf-push unlearning-cleanslate/memorization_eval
        """,
    )

    # Parallel-specific args
    parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU IDs (e.g., '1,2,3,4,5,6')",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merge step after parallel eval",
    )

    # Required eval args
    parser.add_argument("-d", "--hf-dataset", required=True, help="HuggingFace dataset ID")
    parser.add_argument("-c", "--hf-config", required=True, help="Dataset config/split")

    # Optional eval args (with defaults matching cli.py)
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("-b", "--backend", default="sglang", help="Backend: sglang or huggingface")
    parser.add_argument("-n", "--limit", type=int, help="Total number of songs to process")
    parser.add_argument("-w", "--window-size", type=int, default=50, help="Window size in tokens")
    parser.add_argument("--stride", type=int, default=25, help="Stride in tokens")
    parser.add_argument("-t", "--threshold", type=float, default=0.001, help="Memorization threshold")
    parser.add_argument("--no-generate", action="store_true", help="Skip generation")
    parser.add_argument("--generate-all", action="store_true", help="Generate for ALL windows")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--gen-top-k", type=int, default=40, help="Top-k for generation")
    parser.add_argument("--capture-top-k", type=int, default=5, help="Top-k alternatives to save")
    parser.add_argument("-o", "--output", default="results/eval", help="Local output directory")
    parser.add_argument("--hf-push", help="Push merged results to HuggingFace repo")
    parser.add_argument("--text-col", default="reference_target", help="Column containing text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # SGLang memory settings
    parser.add_argument("--mem-fraction", type=float, default=0.85, help="GPU memory fraction for SGLang (default: 0.85)")
    parser.add_argument("--cuda-graph-bs", type=int, default=1, help="Max batch size for CUDA graphs (default: 1, lower = less memory)")

    args = parser.parse_args()

    # Parse GPU list
    gpus = [g.strip() for g in args.gpus.split(",")]
    logger.info(f"Using GPUs: {gpus}")

    # Get dataset size
    total_rows = get_dataset_size(args.hf_dataset, args.hf_config)

    # Create jobs
    jobs = create_jobs(gpus, total_rows, args.limit)
    logger.info(f"Created {len(jobs)} parallel jobs")

    for job in jobs:
        logger.info(f"  {job.part_id}: GPU {job.gpu_id}, offset={job.offset}, limit={job.limit}")

    # Build output directory name with model info
    model_short = args.model.split("/")[-1]  # e.g., "Qwen3-8B" from "Qwen/Qwen3-8B"
    output_dir = f"{args.output}/{model_short}"
    logger.info(f"Output directory: {output_dir}/{args.hf_config}/")

    # Build base args for eval command (excluding offset/limit/part-id)
    base_args = [
        "-d", args.hf_dataset,
        "-c", args.hf_config,
        "-m", args.model,
        "-b", args.backend,
        "-w", str(args.window_size),
        "--stride", str(args.stride),
        "-t", str(args.threshold),
        "--temperature", str(args.temperature),
        "--gen-top-k", str(args.gen_top_k),
        "--capture-top-k", str(args.capture_top_k),
        "-o", output_dir,
        "--text-col", args.text_col,
        "--seed", str(args.seed),
        "--mem-fraction", str(args.mem_fraction),
        "--cuda-graph-bs", str(args.cuda_graph_bs),
    ]
    if args.no_generate:
        base_args.append("--no-generate")
    if args.generate_all:
        base_args.append("--generate-all")

    # Create job queue
    job_queue: queue.Queue[EvalJob] = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    # Track results
    results: dict[str, int] = {}

    # Start worker threads (one per GPU)
    threads: list[threading.Thread] = []
    for _ in gpus:
        t = threading.Thread(target=worker, args=(job_queue, base_args, results), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Check results
    failed = [part_id for part_id, code in results.items() if code != 0]
    if failed:
        logger.error(f"Failed parts: {failed}")
        logger.error("Skipping merge due to failures")
        return

    logger.info("All parallel jobs completed successfully")

    # Run merge
    if not args.no_merge:
        merge_result = run_merge(output_dir, args.hf_config, args.hf_push)
        if merge_result != 0:
            logger.error("Merge failed")
        else:
            logger.info("Merge completed successfully")
    else:
        logger.info("Skipping merge (--no-merge specified)")


if __name__ == "__main__":
    main()
