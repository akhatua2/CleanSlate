import logging
import os
import queue
import subprocess
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SPLITS_REMAINING = [
    "direct__none",
    "direct__first_line",
    "direct__first_verse",
    "direct__half",
    "helpful_assistant__none",
    "helpful_assistant__first_line",
    "helpful_assistant__first_verse",
    "helpful_assistant__half",
    "music_expert__none",
    "music_expert__first_line",
    "music_expert__first_verse",
    "music_expert__half",
    "educational__none",
    "educational__first_line",
    "educational__first_verse",
    "educational__half",
    "transcription__none",
    "transcription__first_line",
    "transcription__first_verse",
    "transcription__half",
    "completion__none",
    "completion__first_line",
    "completion__first_verse",
    "completion__half",
    "karaoke__none",
    "karaoke__first_line",
    "karaoke__first_verse",
    "karaoke__half",
    "quiz__none",
    "quiz__first_line",
    "quiz__first_verse",
    "quiz__half",
    "translation__none",
    "translation__first_line",
    "translation__first_verse",
    "translation__half",
    "analysis__none",
    "analysis__first_line",
    "analysis__first_verse",
    "analysis__half",
    "no_system__none",
    "no_system__first_line",
    "no_system__first_verse",
    "no_system__half",
    "structured_json__none",
    "structured_json__first_line",
    "structured_json__first_verse",
    "structured_json__half",
]


def _parse_gpu_list() -> list[str]:
    return ["6", "7"]
    

def _run_split(gpu_id: str, split_name: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [
        "cleanslate",
        "generate-prompts",
        "--prompts-dataset-id",
        "unlearning-cleanslate/formatted_songs",
        "--split",
        split_name,
        "--model",
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "--backend",
        "sglang",
        "--no-thinking",
        "--max-tokens",
        "8192",
        "--repetition-penalty",
        "1.5",
        "--num-generations",
        "1",
        "--batch-size",
        "128",
        "--capture-top-k",
        "20",
        "--json-decoding",
        "--output-repo",
        "unlearning-cleanslate/generated_songs_qwen3_next_80b_a3b_instruct_json_decoding",
        "--output-config",
        split_name,
    ]

    logger.info("[gpu=%s] start %s", gpu_id, split_name)
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        logger.error(
            "[gpu=%s] failed %s (exit=%s)", gpu_id, split_name, result.returncode
        )
    else:
        logger.info("[gpu=%s] done  %s", gpu_id, split_name)


def _worker(gpu_id: str, work_q: "queue.Queue[str]") -> None:
    while True:
        try:
            split_name = work_q.get_nowait()
        except queue.Empty:
            return
        try:
            _run_split(gpu_id, split_name)
        except Exception:
            logger.exception("[gpu=%s] exception running %s", gpu_id, split_name)
        finally:
            work_q.task_done()


def main() -> None:
    gpus = _parse_gpu_list()
    if not gpus:
        raise RuntimeError("No GPUs available in CUDA_VISIBLE_DEVICES")

    work_q: queue.Queue[str] = queue.Queue()
    for s in SPLITS_REMAINING:
        work_q.put(s)

    threads: list[threading.Thread] = []
    for gpu_id in gpus:
        t = threading.Thread(target=_worker, args=(gpu_id, work_q), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
