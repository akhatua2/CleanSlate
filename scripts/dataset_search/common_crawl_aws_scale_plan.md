# Common Crawl IGM AWS Scale-Up Plan

## Goal

Scale the existing `scripts/dataset_search` per-song infini-gram workflow to one
restartable EC2 run over a prebuilt Common Crawl index, then store all artifacts
in S3.

Primary deliverable: rich per-song artifacts from `igm_batch.py` / `igm_one.py`,
not a compact Hugging Face dataset.

This is a research artifact, not a production service. Prefer failing loudly and
early over silently producing partial or ambiguous results. The only intended
restart behavior is explicit per-song checkpointing via completed artifacts.

Target index:

- Corpus: Common Crawl January 2025 Crawl
- infini-gram index name: `v2_cc_2025-05`
- S3 prefix used for download: `s3://infini-gram-mini/index/v2_cc-2025-05/`
- Local mount/download path: `/mnt/igm-index/v2_cc-2025-05`
- Output S3 prefix: `s3://cleanslate-aws/igm-runs/common-crawl/v2_cc_2025-05/<run_id>/`

## Why Local Index On EC2

The current scripts call `engine.count()`, `engine.find()`, and
`engine.get_doc_by_rank()` many times per song. For each word start position,
`compute_matched_word_ngrams()` grows the phrase until the corpus count drops to
zero. For selected spans, `unique_docs_for_spans()` then fetches matching
documents.

That access pattern is a poor fit for the hosted HTTP API because a full run can
produce millions of tiny count/find/doc requests. A local prebuilt index keeps
the same code path as the 10B run, avoids remote API rate limits, and makes the
run restartable at the per-song artifact level.

Do not build a new FM index on EC2 for this run. Only download the public
prebuilt index.

## EC2 Setup

Use a storage-optimized instance with local NVMe instance store. This workload
does many random mmap-backed reads over a multi-TB FM index, so local NVMe is a
better default than a single large EBS volume.

Recommended default:

- `i4i.8xlarge`
- 32 vCPU, 256 GiB RAM
- 2 x 3.75 TB local NVMe SSD, 7.5 TB total raw instance-store capacity
- Current checked `us-east-1` Linux on-demand price: about `$2.746/hr`

Use `i4i.16xlarge` only if the smoke test is clearly CPU-bound or if the output
artifacts need more local scratch:

- 64 vCPU, 512 GiB RAM
- 4 x 3.75 TB local NVMe SSD, 15 TB total raw instance-store capacity
- Current checked `us-east-1` Linux on-demand price: about `$5.491/hr`

Avoid the initial EBS-only plan unless you need persistence across stop/start.
For comparison, `r8i.8xlarge` is about `$2.223/hr` before storage, and a 6 TB
`gp3` volume in `us-east-1` is about `$480/month` prorated before any extra
provisioned IOPS/throughput. That makes EBS-only roughly similar in hourly cost
for a short run, but slower and more tuning-sensitive for random index reads.

Important instance-store tradeoff: local NVMe is lost when the instance stops or
terminates. Treat S3 as the durable checkpoint and sync outputs regularly.

Minimum run settings:

- mmap mode for the index; do not pass `--load-to-ram`
- Start with low parallelism: `--song-workers 1 --workers 4`
- Increase only after a smoke test confirms page-cache and disk behavior are
  stable

The January 2025 Common Crawl index is several TB. Keep enough headroom for
partial downloads, logs, parquet outputs, and resumed runs.

Use on-demand for the first full run. Spot can be cheaper, but interruption on
instance-store means losing the local index copy and any unsynced outputs. If
using Spot anyway, sync outputs to S3 frequently and expect to re-download the
index after interruption.

Launch in `us-east-1` unless you have a verified reason not to. The
`cleanslate-aws` bucket is in the classic `LocationConstraint: null` region,
which means `us-east-1`. Same-region S3 to AWS-service data transfer avoids the
large internet data-transfer line item; a different region can introduce
cross-region transfer costs.

Add an S3 gateway VPC endpoint if running in a private subnet. Otherwise S3
traffic through a NAT Gateway can add avoidable per-GB processing charges.

Do not mount S3 as the live index filesystem. Sync the index once to local NVMe,
then run against `/mnt/igm-index/v2_cc-2025-05`.

Format the two NVMe devices on `i4i.8xlarge` as a single scratch mount:

```bash
sudo mdadm --create /dev/md0 --level=0 --raid-devices=2 /dev/nvme1n1 /dev/nvme2n1
sudo mkfs.xfs -f /dev/md0
sudo mkdir -p /mnt
sudo mount /dev/md0 /mnt
sudo chown "$USER":"$USER" /mnt
df -h /mnt
```

For `i4i.16xlarge`, use `--raid-devices=4` and include all four instance-store
NVMe devices.

## Download The Index

The public infini-gram bucket is requester-pays, so include
`--request-payer requester`.

Use shard-parallel download instead of one monolithic sync. Sync all top-level
index shards together so the long download is dominated by available network and
NVMe throughput rather than single-process scheduling. This is intentionally
aggressive for a one-shot research run; fail loudly if any shard sync fails.

```bash
set -euo pipefail

mkdir -p /mnt/igm-index/v2_cc-2025-05

aws s3 ls s3://infini-gram-mini/index/v2_cc-2025-05/ \
    --request-payer requester \
    | awk '/PRE/ {gsub("/", "", $2); print $2}' \
    | sort \
    > /mnt/igm-index/v2_cc-2025-05.shards.txt

cat /mnt/igm-index/v2_cc-2025-05.shards.txt

N_SHARDS=$(wc -l < /mnt/igm-index/v2_cc-2025-05.shards.txt)
test "$N_SHARDS" -gt 0

cat /mnt/igm-index/v2_cc-2025-05.shards.txt \
    | xargs -P "$N_SHARDS" -I {} bash -c '
        set -euo pipefail
        shard="$1"
        mkdir -p "/mnt/igm-index/v2_cc-2025-05/$shard"
        aws s3 sync \
            "s3://infini-gram-mini/index/v2_cc-2025-05/$shard/" \
            "/mnt/igm-index/v2_cc-2025-05/$shard/" \
            --request-payer requester
      ' _ {}
```

Validate the layout before running search:

```bash
set -euo pipefail

find /mnt/igm-index/v2_cc-2025-05 -maxdepth 2 -type f | head

test "$(find /mnt/igm-index/v2_cc-2025-05 -name data.fm9 | wc -l)" -gt 0
test "$(find /mnt/igm-index/v2_cc-2025-05 -name meta.fm9 | wc -l)" -gt 0
test "$(find /mnt/igm-index/v2_cc-2025-05 -name data_offset | wc -l)" -gt 0
test "$(find /mnt/igm-index/v2_cc-2025-05 -name meta_offset | wc -l)" -gt 0

du -sh /mnt/igm-index/v2_cc-2025-05
```

Expected shape: numeric shard directories such as `00/`, `01/`, ... with
`data.fm9`, `meta.fm9`, `data_offset`, and `meta_offset`. The existing
`load_engine()` code supports numeric shard directories.

If any shard sync fails, the `xargs` command exits non-zero. Re-run the same
command after fixing the cause; completed shard directories should be skipped by
`aws s3 sync`.

## Smoke Test

Use two smoke tiers:

1. Correctness smoke: `Never Gonna Give You Up` plus optionally one book-scale
   item. This catches engine/load/artifact failures quickly.
2. Stress smoke: all six IDs below. This exercises the long-reference path and
   is closer to the expensive full-run behavior.

Candidate smoke IDs:

| song_id | title | creator | year | memorized_fraction | size |
| --- | --- | --- | --- | ---: | ---: |
| `b9bb555ab4913dfeaafb409555e8f543f5180fc6` | Never Gonna Give You Up | Rick Astley | 1988 | 0.176829 | 1,839 chars / 362 words |
| `731bc81c4a7cd705342bb30f8b6d9f4a5d658d04` | A Song of Ice and Fire (Complete) | George R.R. Martin | 1996 | 0.000451 | 7,388,860 chars / 1,356,337 words |
| `225cde9fb93963bee1a264bcc6d296b6361374e0` | The Complete Novels of George Orwell | George Orwell | 1949 | 0.005339 | 2,725,430 chars / 488,489 words |
| `6badee68f339d8e8b416b69619ef8b80d56f42b5` | War and Peace | Leo Tolstoy | 1869 | 0.000714 | 2,591,434 chars / 455,605 words |
| `48d98b0f0b7d5bc7bc9499c798a5c337af37afc9` | The Lord of the Rings | J.R.R. Tolkien | 1954 | 0.002572 | 2,527,491 chars / 475,818 words |
| `4d50a24058da021d176e81365370ffadbf06a6ed` | The Maze Runner Series | James Dashner | 2009 | 0.000851 | 1,598,720 chars / 286,958 words |

Write the stress smoke ID file:

```bash
cat > /mnt/smoke_song_ids.json <<'JSON'
[
  "b9bb555ab4913dfeaafb409555e8f543f5180fc6",
  "731bc81c4a7cd705342bb30f8b6d9f4a5d658d04",
  "225cde9fb93963bee1a264bcc6d296b6361374e0",
  "6badee68f339d8e8b416b69619ef8b80d56f42b5",
  "48d98b0f0b7d5bc7bc9499c798a5c337af37afc9",
  "4d50a24058da021d176e81365370ffadbf06a6ed"
]
JSON
```

Run the smoke batch:

```bash
set -euo pipefail

RUN_ID=smoke-$(date -u +%Y%m%dT%H%M%SZ)
OUT=/mnt/igm-runs/common-crawl-v2_cc_2025-05/$RUN_ID

python scripts/dataset_search/igm_batch.py \
    --song-ids /mnt/smoke_song_ids.json \
    --index-dir /mnt/igm-index/v2_cc-2025-05 \
    --output-dir "$OUT" \
    --workers 4 \
    --song-workers 1
```

Check the output:

```bash
set -euo pipefail

find "$OUT" -maxdepth 3 -type f | head -100
python -m py_compile scripts/dataset_search/igm_batch.py scripts/dataset_search/igm_one.py

test -s "$OUT/run_meta.json"
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
meta = json.loads((out / "run_meta.json").read_text())
bad = {
    sid: s for sid, s in meta.get("summaries", {}).items()
    if s.get("data_status") != "ok" or s.get("n_rows", 0) <= 0
}
if bad:
    raise SystemExit(f"smoke test failed or returned empty docs: {bad}")
print("smoke summaries ok")
PY
```

For each successful song, expect:

- `summary.json`
- `meta.json`
- `reference_text.txt`
- `matches.parquet`
- `selected_spans.parquet`
- `docs.parquet`
- `occ_counts.npz`
- `occ_strips.png`

If the smoke test is slow but stable, keep the conservative settings. If disk
I/O and memory are healthy, try `--workers 8` or `--song-workers 2`, but avoid
large jumps. Each song subprocess loads the same multi-TB mmap-backed index.

## Full Run

Use a timestamped run id and keep all state local until the run is either done
or intentionally paused. The original 10B workflow selected items by
memorized-fraction threshold; use `--memfraction-threshold 0.0` to run every
row in the default OLMo verification dataset. Use a higher threshold such as
`0.05` only when intentionally restricting to the memorized subset.

```bash
set -euo pipefail

RUN_ID=full-$(date -u +%Y%m%dT%H%M%SZ)
OUT=/mnt/igm-runs/common-crawl-v2_cc_2025-05/$RUN_ID

python scripts/dataset_search/igm_batch.py \
    --memfraction-threshold 0.0 \
    --index-dir /mnt/igm-index/v2_cc-2025-05 \
    --output-dir "$OUT" \
    --workers 4 \
    --song-workers 1
```

Restart behavior:

- `igm_batch.py` writes each song under `songs/<song_id>/`.
- `igm_one.py` treats an existing `docs.parquet` as cached unless
  `--overwrite` is passed.
- Re-run the same command to resume failed/interrupted work.
- Logs live under `_logs/`.
- Final run metadata lives in `run_meta.json`.

Do not use `--load-to-ram` for this index unless the EC2 instance has enough RAM
for the full multi-TB index and you have explicitly budgeted for it.

## Post-Release Overlap Experiment

Goal: search content released after the target indexed corpus and inspect any
overlap with the January 2025 Common Crawl index. Treat nonzero overlap as a
research signal to inspect, not as proof of train/test leakage: Common Crawl may
contain previews, announcements, quoted excerpts, mislabeled dates, syndicated
lyrics pages, or later crawled pages depending on the exact index snapshot.

Inputs:

- `unlearning-cleanslate/cleanslate_dataset_post_llama31`
- `unlearning-cleanslate/cleanslate_dataset_post_books_2025`

Schema check:

- `cleanslate_dataset_post_llama31`: 158 rows, years 2025-2026, about 276k chars
  / 55k words total.
- `cleanslate_dataset_post_books_2025`: 3 rows, year 2025, about 240k chars /
  43k words total.
- Both use `content_id`, `content_title`, `content_creators`, `content_year`,
  `reference_target`.
- `igm_batch.py` now supports generic dataset columns via `--id-col`,
  `--title-col`, `--creators-col`, and `--year-col`, while preserving the
  original OLMo defaults.

Plan:

1. Validate the post-release workflow locally against the existing 10B index.
2. Use explicit ID files for reproducibility; do not rely on row-order-only
   `--all-rows` for paid EC2 runs.
3. Snapshot slim normalized Parquet inputs and ID files to S3 for provenance.
4. Reuse the same `igm_batch.py` command shape on EC2 with the Common Crawl
   index and a different output prefix.

Do not upload the full OLMo verification dataset with all nested eval columns.
For retrieval runs, only snapshot the slim fields needed for search and
provenance: IDs, title, creator, year, `reference_target`, optional
`memorized_fraction`, source dataset, and source split.

### Local 10B Validation

Use the local 10B FM index first:

- Local 10B index: `curation/indices/10B/infini_gram_mini/b6c7e0c77ec6`
- Output root: `results/igm_post_release_10b/<run_id>/`
- Dataset IDs:
  - `unlearning-cleanslate/cleanslate_dataset_post_llama31`
  - `unlearning-cleanslate/cleanslate_dataset_post_books_2025`

Create explicit local ID files:

```bash
set -euo pipefail

scripts/dataset_search/make_post_release_inputs.sh
```

Run one dataset locally against the 10B index:

```bash
set -euo pipefail

RUN_ID=local-10b-$(date -u +%Y%m%dT%H%M%SZ)
OUT=results/igm_post_release_10b/post_books_2025/$RUN_ID

.venv/bin/python scripts/dataset_search/igm_batch.py \
    --dataset unlearning-cleanslate/cleanslate_dataset_post_books_2025 \
    --dataset-config default \
    --dataset-split train \
    --id-col content_id \
    --title-col content_title \
    --creators-col content_creators \
    --year-col content_year \
    --text-col reference_target \
    --song-ids results/igm_post_release_inputs/post_books_2025/ids.json \
    --output-dir "$OUT" \
    --song-workers 3 \
    --workers 8
```

Check local 10B outputs:

```bash
set -euo pipefail

test -s "$OUT/run_meta.json"
.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
meta = json.loads((out / "run_meta.json").read_text())
bad = {
    cid: s for cid, s in meta.get("summaries", {}).items()
    if s.get("returncode", 0) != 0 or s.get("data_status") not in {"ok", "cached"}
}
if bad:
    raise SystemExit(f"bad post-release summaries: {bad}")
print(json.dumps(meta["status_counts"], indent=2))
PY
```

If that passes, run the lyrics-scale post-release dataset:

```bash
set -euo pipefail

RUN_ID=local-10b-$(date -u +%Y%m%dT%H%M%SZ)
OUT=results/igm_post_release_10b/post_llama31/$RUN_ID

.venv/bin/python scripts/dataset_search/igm_batch.py \
    --dataset unlearning-cleanslate/cleanslate_dataset_post_llama31 \
    --dataset-config default \
    --dataset-split train \
    --id-col content_id \
    --title-col content_title \
    --creators-col content_creators \
    --year-col content_year \
    --text-col reference_target \
    --song-ids results/igm_post_release_inputs/post_llama31/ids.json \
    --output-dir "$OUT" \
    --song-workers 16 \
    --workers 16
```

Check it the same way:

```bash
set -euo pipefail

test -s "$OUT/run_meta.json"
.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
meta = json.loads((out / "run_meta.json").read_text())
bad = {
    cid: s for cid, s in meta.get("summaries", {}).items()
    if s.get("returncode", 0) != 0 or s.get("data_status") not in {"ok", "cached"}
}
if bad:
    raise SystemExit(f"bad post-release summaries: {bad}")
print(json.dumps(meta["status_counts"], indent=2))
PY
```

For post-release overlap, zero retrieved docs is a valid result. Fail on
subprocess errors or bad status, not on `n_rows == 0`.

### Common Crawl Post-Release Run

After the local 10B validation passes, upload the input ID files and slim
metadata, then use the same `igm_batch.py` flags on EC2 with the Common Crawl
index:

```bash
set -euo pipefail

aws s3 sync \
    results/igm_post_release_inputs/ \
    s3://cleanslate-aws/igm-inputs/post_release/
```

On EC2, after downloading the Common Crawl index, sync the input files and run
the post-release datasets to a dedicated output root:

```bash
set -euo pipefail

aws s3 sync \
    s3://cleanslate-aws/igm-inputs/post_release/ \
    /mnt/igm-inputs/post_release/

RUN_ID=post-release-cc-$(date -u +%Y%m%dT%H%M%SZ)
INDEX=/mnt/igm-index/v2_cc-2025-05
OUT_ROOT=/mnt/igm-runs/common-crawl-v2_cc_2025-05/post_release/$RUN_ID

.venv/bin/python scripts/dataset_search/igm_batch.py \
    --dataset unlearning-cleanslate/cleanslate_dataset_post_books_2025 \
    --dataset-config default \
    --dataset-split train \
    --id-col content_id \
    --title-col content_title \
    --creators-col content_creators \
    --year-col content_year \
    --text-col reference_target \
    --song-ids /mnt/igm-inputs/post_release/post_books_2025/ids.json \
    --index-dir "$INDEX" \
    --output-dir "$OUT_ROOT/post_books_2025" \
    --song-workers 3 \
    --workers 8

.venv/bin/python scripts/dataset_search/igm_batch.py \
    --dataset unlearning-cleanslate/cleanslate_dataset_post_llama31 \
    --dataset-config default \
    --dataset-split train \
    --id-col content_id \
    --title-col content_title \
    --creators-col content_creators \
    --year-col content_year \
    --text-col reference_target \
    --song-ids /mnt/igm-inputs/post_release/post_llama31/ids.json \
    --index-dir "$INDEX" \
    --output-dir "$OUT_ROOT/post_llama31" \
    --song-workers 16 \
    --workers 16
```

Sync post-release outputs separately:

```bash
set -euo pipefail

DEST=s3://cleanslate-aws/igm-runs/common-crawl/v2_cc_2025-05/post_release/$RUN_ID/

aws s3 sync "$OUT_ROOT/" "$DEST"

LOCAL_COUNT=$(find "$OUT_ROOT" -type f | wc -l)
REMOTE_COUNT=$(aws s3 ls "$DEST" --recursive | awk 'NF >= 4 {n += 1} END {print n + 0}')
test "$LOCAL_COUNT" -eq "$REMOTE_COUNT"
```

## S3 Artifact Sync

Sync to the CleanSlate bucket after the smoke test and periodically during the
full run if the instance is at risk of interruption.

```bash
set -euo pipefail

DEST=s3://cleanslate-aws/igm-runs/common-crawl/v2_cc_2025-05/$RUN_ID/

aws s3 sync "$OUT/" "$DEST"
```

After sync, compare local and remote summaries:

```bash
set -euo pipefail

find "$OUT" -type f | wc -l
du -sh "$OUT"

aws s3 ls "$DEST" --recursive --summarize
```

Do not delete local outputs until S3 object counts and total size look
reasonable.

For the final sync, fail if the local and remote file counts differ:

```bash
set -euo pipefail

LOCAL_COUNT=$(find "$OUT" -type f | wc -l)
REMOTE_COUNT=$(aws s3 ls "$DEST" --recursive | awk 'NF >= 4 {n += 1} END {print n + 0}')
test "$LOCAL_COUNT" -eq "$REMOTE_COUNT"
```

## Suggested Manifest

Before the final sync, write a small manifest into the output directory:

```json
{
  "run_id": "<run_id>",
  "index_name": "v2_cc_2025-05",
  "index_s3": "s3://infini-gram-mini/index/v2_cc-2025-05/",
  "index_local": "/mnt/igm-index/v2_cc-2025-05",
  "output_s3": "s3://cleanslate-aws/igm-runs/common-crawl/v2_cc_2025-05/<run_id>/",
  "script": "scripts/dataset_search/igm_batch.py",
  "primary_params": {
    "retrieval_knobs": "script defaults",
    "workers": 4,
    "song_workers": 1,
    "load_to_ram": false
  }
}
```

The existing `run_meta.json` records most run details automatically; the
manifest is for quick S3 browsing and provenance.

## Operational Notes

- Fail fast on missing index files, empty `docs.parquet`, failed smoke summaries,
  failed S3 syncs, or S3/local count mismatches.
- Do not mask subprocess failures with `|| true`; inspect `_logs/<song_id>.log`
  and rerun explicitly after fixing the cause.
- Common Crawl metadata may not look like the Dolma `path` metadata used in the
  10B run. Treat missing or unusual `path` values as acceptable if `doc_ix` and
  `doc_content` are present.
- Do not override retrieval knobs (`min_n`, `n_values`, `top_k_spans`,
  `docs_per_span`, etc.) for the official run unless there is an explicit
  research reason. The script defaults are the intended configuration.
- The most important checkpoint unit is a completed
  `songs/<song_id>/docs.parquet`.
- Keep the run simple: one EC2 instance, one local index copy, one output
  directory, periodic S3 syncs.

## Acceptance Criteria

- The index is downloaded and validates as a multi-shard infini-gram mini index.
- A smoke batch produces complete per-song artifacts.
- The full run can be interrupted and resumed without recomputing completed
  songs.
- Final artifacts are available under
  `s3://cleanslate-aws/igm-runs/common-crawl/v2_cc_2025-05/<run_id>/`.
- `run_meta.json` and the manifest identify the exact index, parameters, and
  output location.
