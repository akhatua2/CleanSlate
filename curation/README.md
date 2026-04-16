# curation/

Retrieval pipeline for forget-set curation. For every item in `unlearning-cleanslate/cleanslate_dataset`, searches for verbatim matches of `reference_target` in Dolma3, following the OLMoTrace methodology (arxiv 2504.07096).

## Methods

| Method | Backend | Index size | Query cost |
|--------|---------|-----------|------------|
| `infini_gram_mini` | FM-index (exact n-gram, byte-level) | ~0.44x text size | ~50ms per `count()` |
| `bm25` | Sharded `bm25s` | ~1-2x text size | top-k per shard, merged with heap |

## Layout

```
curation/
  retrieval/
    schema.py              Cfg dataclass, MATCH_SCHEMA, HF_FEATURES
    queries.py             load_queries() -> pl.DataFrame
    backend_infini_gram.py build_fm_index(), search_one_item()
    backend_bm25.py        segment_shard(), build_bm25_index(), search_bm25_one_item()
    stages.py              search(), export(), run_all() — idempotent stages
    cli.py                 unified CLI
  vendor/infini-gram-mini/  Cloned from github.com/xuhaoxh/infini-gram-mini
  indices/{variant}/{method}/{hash}/    FM-index or BM25 shards
  outputs/{method}_cleanslate_dolma3_{variant}/{hash}/{items/,matches.parquet,summary.parquet}
```

## Design Principles

- **Idempotent stages**: each stage (`build_index`, `search`, `export`) is safe to re-run; "done" is defined by output existence. Per-item parquets are the restart granularity.
- **Streaming**: zstd decompression via `io.TextIOWrapper`, batched parquet writes with `ParquetWriter`. No `.collect()` / `.to_pylist()` on full shards.
- **Schema-enforced output**: hits are plain dicts validated by `MATCH_SCHEMA` at write time. No dataclass intermediary.
- **One `Cfg` object** instead of separate corpus + query specs. Hashed into `index_hash` / `query_hash` so incompatible configs can't accidentally reuse each other's outputs.

## HF Datasets (published)

- `unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B` — 287K hits over 4666 items, 10B Dolma3 (FM-index exact match)
- `unlearning-cleanslate/verif-bm25-dolma-10B` — BM25 top-k retrieval on 10B Dolma3
- (6T pending — requires 40TB+ disk for SA construction)

## Setup

### 1. Clone + install

```bash
git clone <repo-url> /path/to/CleanSlate && cd /path/to/CleanSlate
git checkout fix/char-based-windowing
uv venv --python 3.12
uv pip install -e ".[cli,export,unlearning,eval]" polars pybind11 zstandard bm25s
```

### 2. Clone vendored infini-gram-mini

`curation/vendor/` is gitignored. Clone the upstream repo there on every fresh checkout:

```bash
git clone https://github.com/xuhaoxh/infini-gram-mini.git \
    curation/vendor/infini-gram-mini
```

This ships pre-compiled `sdsl`/`divsufsort` static libs under `sdsl/lib/` and a `rust_indexing` binary. Only the C++ engine and `cpp_indexing` need compiling against the local Python ABI and glibc — see step 3.

### 3. Compile infini-gram-mini

The vendored infini-gram-mini ships pre-compiled `rust_indexing` + `cpp_indexing` binaries, but:
- **`cpp_indexing` requires `libcilkrts.so.5` which is not on modern Linux.** Recompile `cpp_indexing_local` without Cilk.
- **Python C++ engine must be compiled against your Python 3.12 ABI.**

```bash
cd curation/vendor/infini-gram-mini/engine
PY=$(pwd)/../../../../.venv/bin/python
c++ -std=c++17 -O3 -shared -fPIC $($PY -m pybind11 --includes) \
    src/cpp_engine.cpp \
    -o src/cpp_engine$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))") \
    -I../sdsl/include -L../sdsl/lib \
    -lsdsl -ldivsufsort -ldivsufsort64 -pthread

cd ../src
g++ -std=c++17 -O3 -no-pie \
    -I../sdsl/include -L../sdsl/lib \
    indexing.cpp -o cpp_indexing_local \
    -lsdsl -ldivsufsort -ldivsufsort64 -pthread

cd ../../../..
```

Verify:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'curation/vendor/infini-gram-mini/engine')
from src.engine import InfiniGramMiniEngine
print('OK')
"
```

### 4. Dolma3 corpus data

Expected layout: `$DOLMA_ROOT/raw/{10B,6T}/data/<category>/<shard>.jsonl.zst`

- **10B** (`allenai/dolma3_dolmino_mix-10B-1025`): 18GB compressed, 1113 shards, 165 categories — curated annealing subset for OLMo-3 stage 2.
- **6T** (`allenai/dolma3_mix-6T`): full pretraining corpus, 63 categories across CC, olmocr PDFs, stack_edu code, finemath, wiki, arxiv. Full download is >6TB compressed.
- **6T local subset**: 5 CC categories only (`entertainment`, `literature`, `art_and_design`, `history_and_geography`, `religion`), 624GB compressed, 7105 shards.

#### Download 10B (full)

```bash
export DOLMA_ROOT=/path/to/shared_hf_cache/dolma3
export HF_TOKEN=<your-token>   # dolma3 datasets are gated

mkdir -p $DOLMA_ROOT/raw/10B
huggingface-cli download allenai/dolma3_dolmino_mix-10B-1025 \
    --repo-type dataset \
    --local-dir $DOLMA_ROOT/raw/10B

# Dataset ships with data/ subdir — no rearranging needed.
# Verify:
find $DOLMA_ROOT/raw/10B/data -name "*.jsonl.zst" | wc -l   # should be ~1113
```

#### Download 6T (selected categories)

Full 6T is >6TB. Pick categories relevant to your query set:

```bash
export DOLMA_ROOT=/path/to/shared_hf_cache/dolma3
export HF_TOKEN=<your-token>

mkdir -p $DOLMA_ROOT/raw/6T

# Option A: match the existing local subset (5 CC categories, ~624GB)
huggingface-cli download allenai/dolma3_mix-6T \
    --repo-type dataset \
    --include \
        "data/common_crawl-entertainment-*/*" \
        "data/common_crawl-literature-*/*" \
        "data/common_crawl-art_and_design-*/*" \
        "data/common_crawl-history_and_geography-*/*" \
        "data/common_crawl-religion-*/*" \
    --local-dir $DOLMA_ROOT/raw/6T

# Option B: just 2 categories likely to contain lyrics/book text (~350GB)
huggingface-cli download allenai/dolma3_mix-6T \
    --repo-type dataset \
    --include \
        "data/common_crawl-entertainment-*/*" \
        "data/common_crawl-literature-*/*" \
    --local-dir $DOLMA_ROOT/raw/6T

# Option C: full 6T (requires ~7TB disk for raw data alone)
huggingface-cli download allenai/dolma3_mix-6T \
    --repo-type dataset \
    --local-dir $DOLMA_ROOT/raw/6T

# The `corpora.Cfg.raw_shards()` glob picks up files anywhere under
# $DOLMA_ROOT/raw/6T/ (prefers raw/6T/data/ if it exists) — no rearranging needed.
# Verify:
find $DOLMA_ROOT/raw/6T -name "*.jsonl.zst" | wc -l
```

#### List all 6T categories (for reference)

```python
from huggingface_hub import HfApi
api = HfApi()
folders = api.list_repo_tree("allenai/dolma3_mix-6T", repo_type="dataset", path_in_repo="data", recursive=False)
cats = sorted({f.path.rsplit("-", 1)[0].replace("data/", "") for f in folders if "-" in f.path})
for c in cats: print(c)
```

Produces 63 categories: 24 CC (adult_content, art_and_design, crime_and_law, education_and_jobs, electronics_and_hardware, entertainment, fashion_and_beauty, finance_and_business, food_and_dining, games, health, history_and_geography, home_and_hobbies, industrial, literature, politics, religion, science_math_and_technology, social_life, software, software_development, sports_and_fitness, transportation, travel_and_tourism), 18 olmocr_science_pdfs variants, 15 stack_edu languages, plus `finemath-3plus`, `dolma1_7-wiki-en`, `rpj-proofpile-arxiv`.

## Running

The CLI has 4 subcommands: `build-index`, `search`, `export`, `run` (= build→search→export).

### 10B end-to-end

```bash
.venv/bin/python -m curation.retrieval.cli run \
    --method infini_gram_mini --corpus 10B \
    --workers 128 \
    --push-to-hub --hf-repo unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_10B
```

Timing (this machine, 224 CPUs): index build ~90 min, search ~5 hours with doc retrieval for every span.

### 6T end-to-end

With `--num-index-shards N`, the FM-index is built as N independent shards sequentially, deleting intermediates between shards. `InfiniGramMiniEngine` natively unions query results across shard dirs, so search behaves identically to a monolithic index (modulo sharded query latency).

**Disk requirement (recommended: `--num-index-shards 4`):**
- Per shard (1/N of corpus): text ~1TB + SA ~1TB + BWT ~1TB + temp ~1TB = ~4TB peak transient
- Intermediates deleted after each shard's `.fm9` is written
- Final on-disk: ~N × 750MB per shard = **~3TB total**

With `--num-index-shards 1` (monolithic, old behavior):
- Step 1 prepare: text_data.sdsl ~4.1TB + text_meta.sdsl ~1.3TB
- Step 2 SA+BWT: ~15TB peak transient
- Final: ~11TB
- **Requires ~40TB peak disk** — not recommended.

```bash
export DOLMA=/path/to/shared_hf_cache/dolma3

# Full 5-category local subset, sharded into 4 to fit ~5TB disks
.venv/bin/python -m curation.retrieval.cli run \
    --method infini_gram_mini --corpus 6T \
    --num-index-shards 4 \
    --workers 128 --index-workers 64 --index-mem-gb 512 \
    --dolma-dir $DOLMA \
    --push-to-hub --hf-repo unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_6T

# Or restrict to specific categories to fit smaller disks (can use fewer shards)
.venv/bin/python -m curation.retrieval.cli run \
    --method infini_gram_mini --corpus 6T \
    --categories common_crawl-entertainment,common_crawl-literature \
    --num-index-shards 2 \
    --workers 128 --index-workers 64 --index-mem-gb 512 \
    --dolma-dir $DOLMA \
    --push-to-hub --hf-repo unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_6T
```

Timing estimate (single node, 128 workers): prepare ~4 hours, SA+BWT ~20-30 hours, FM9 ~10 hours, search ~20-40 hours (varies with doc retrieval depth).

### Running in stages

Useful for long jobs — run steps separately with checkpointing between them:

```bash
# Step 1: build index (restartable — skips if data.fm9 exists)
.venv/bin/python -m curation.retrieval.cli build-index \
    --method infini_gram_mini --corpus 6T \
    --workers 64 --index-mem-gb 512

# Step 2: search (restartable — skips items with existing per-item parquet)
.venv/bin/python -m curation.retrieval.cli search \
    --method infini_gram_mini --corpus 6T --workers 128

# Step 3: export + push
.venv/bin/python -m curation.retrieval.cli export \
    --method infini_gram_mini --corpus 6T \
    --push-to-hub --hf-repo unlearning-cleanslate/infini_gram_mini_cleanslate_dolma3_6T
```

### BM25

Same CLI, `--method bm25`. Produces segment parquets + per-shard `bm25s` indices.

```bash
.venv/bin/python -m curation.retrieval.cli run \
    --method bm25 --corpus 10B \
    --workers 128 --num-shards 32 \
    --push-to-hub --hf-repo unlearning-cleanslate/bm25_cleanslate_dolma3_10B
```

### BM25 6T on a larger box

If a `6T` BM25 run died with `No space left on device`, move the run state to a larger scratch volume and resume in stages. The BM25 pipeline is restartable at three levels:

- `build-index` skips existing raw-shard segment parquets under `indices/.../segments/seg_*.parquet`
- `build-index` skips completed shard indices when `bm25_shards/shard_*/data.csc.index.npy` already exists
- `search` skips completed query items when `outputs/.../items/<content_id>.parquet` already exists

Use a dedicated `--curation-root` on the larger box. Do not write `6T` BM25 artifacts into the repo checkout if the local disk is tight.

#### Fresh start on the new box

If you are **not** reusing any partial BM25 artifacts, the move is simpler:

1. Copy Dolma into the new shared location.
2. Pick a large scratch directory for `--curation-root`.
3. Run a fresh `build-index`, `search`, and `export` there.

In that case you do **not** need to preserve the old `index_hash` or `query_hash`, and changing `--dolma-dir` is fine. The new machine will build a new hashed index/output tree from scratch.

Example:

```bash
export DOLMA=/new/shared/dolma3
export CURATION_ROOT=/mnt/nvme/cleanslate-curation

.venv/bin/python -m curation.retrieval.cli run \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --index-workers 8 --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200 \
    --push-to-hub --hf-repo unlearning-cleanslate/bm25_cleanslate_dolma3_6T
```

If you want to stage it explicitly:

```bash
.venv/bin/python -m curation.retrieval.cli build-index \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --index-workers 8 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200

.venv/bin/python -m curation.retrieval.cli search \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200

.venv/bin/python -m curation.retrieval.cli export \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200 \
    --push-to-hub --hf-repo unlearning-cleanslate/bm25_cleanslate_dolma3_6T
```

#### 1. Keep the same hash inputs if you want to resume

Resume works by landing in the same hashed directories. For BM25 that means keeping these inputs unchanged:

- `--method bm25`
- `--corpus 6T`
- `--segment-chars`
- `--segment-overlap`
- `--min-doc-chars`
- `--min-segment-chars`
- `--categories`
- `--num-shards`
- `--top-k` if you also want to reuse query outputs
- dataset source (`--dataset-path` or the default HF dataset)
- `--dolma-dir`

Important: `--dolma-dir` is part of `index_hash`. If you move Dolma to a different absolute path, the code will compute a different `index_dir` and will not see the old partial BM25 index. On the new box, either keep the same Dolma path or recreate that path with a bind mount/symlink.

You can print the exact hashed paths before copying anything:

```bash
.venv/bin/python - <<'PY'
from curation.retrieval.schema import Cfg

cfg = Cfg(
    variant="6T",
    method="bm25",
    top_k=100,
    segment_chars=2000,
    segment_overlap=500,
    min_doc_chars=100,
    min_segment_chars=200,
    num_shards=512,
    dolma_root="/lfs/skampere2/0/shared_hf_cache/dolma3",
)
print("index_dir =", cfg.index_dir)
print("output_dir =", cfg.output_dir)
PY
```

#### 2. Copy the partial run state

If you want to resume instead of restarting from scratch, copy the hashed index and output directories into the new `--curation-root`:

```bash
export OLD_CURATION=/old/path/to/CleanSlate/curation
export NEW_CURATION=/mnt/nvme/cleanslate-curation

mkdir -p "$NEW_CURATION"/indices/6T/bm25
mkdir -p "$NEW_CURATION"/outputs/bm25_cleanslate_dolma3_6T

rsync -a "$OLD_CURATION"/indices/6T/bm25/<index_hash>/ \
    "$NEW_CURATION"/indices/6T/bm25/<index_hash>/

rsync -a "$OLD_CURATION"/outputs/bm25_cleanslate_dolma3_6T/<query_hash>/ \
    "$NEW_CURATION"/outputs/bm25_cleanslate_dolma3_6T/<query_hash>/
```

If the old run failed before `search` started, copying the `outputs/.../<query_hash>/` tree is unnecessary.

#### 3. Resume by stage

Build or resume the BM25 index:

```bash
export DOLMA=/lfs/skampere2/0/shared_hf_cache/dolma3
export CURATION_ROOT=/mnt/nvme/cleanslate-curation

.venv/bin/python -m curation.retrieval.cli build-index \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --index-workers 8 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200
```

Then resume search:

```bash
.venv/bin/python -m curation.retrieval.cli search \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200
```

Then export and push:

```bash
.venv/bin/python -m curation.retrieval.cli export \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200 \
    --push-to-hub --hf-repo unlearning-cleanslate/bm25_cleanslate_dolma3_6T
```

Or rerun the full pipeline once the state is copied:

```bash
.venv/bin/python -m curation.retrieval.cli run \
    --method bm25 --corpus 6T \
    --dolma-dir "$DOLMA" \
    --curation-root "$CURATION_ROOT" \
    --workers 32 --index-workers 8 --top-k 100 \
    --num-shards 512 \
    --segment-chars 2000 --segment-overlap 500 \
    --min-doc-chars 100 --min-segment-chars 200 \
    --push-to-hub --hf-repo unlearning-cleanslate/bm25_cleanslate_dolma3_6T
```

#### 4. Expected disk layout

On the larger box, BM25 artifacts land under:

- `indices/6T/bm25/<index_hash>/segments/seg_*.parquet`
- `indices/6T/bm25/<index_hash>/segments/shard_*.parquet`
- `indices/6T/bm25/<index_hash>/bm25_shards/shard_*/`
- `outputs/bm25_cleanslate_dolma3_6T/<query_hash>/items/*.parquet`
- `outputs/bm25_cleanslate_dolma3_6T/<query_hash>/matches.parquet`
- `outputs/bm25_cleanslate_dolma3_6T/<query_hash>/summary.parquet`

If disk usage is still too high, the only real levers are reducing the corpus with `--categories`, reducing `--num-shards`, or not running BM25 over the full `6T` subset.

## Key Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--method` | `infini_gram_mini` | `infini_gram_mini` or `bm25` |
| `--corpus` | `10B` | Dolma3 variant dir under `raw/` |
| `--workers` | 32 | Parallel worker processes for search |
| `--index-workers` | = workers | Parallel workers for index build |
| `--top-k` | 100 | Spans/hits kept per item |
| `--min-span-chars` | 20 | Minimum verbatim span length (infini-gram). Higher = faster search. |
| `--segment-chars` | 2000 | BM25 segment length |
| `--segment-overlap` | 500 | BM25 segment overlap |
| `--categories` | all | Comma-separated substrings to filter raw shard dirs |
| `--index-mem-gb` | 256 | Memory for SA+BWT construction |
| `--num-shards` | auto | BM25 shard count (10B→32, 6T→512) |
| `--dolma-dir` | `/lfs/skampere2/0/shared_hf_cache/dolma3` | Override Dolma3 root |
| `--curation-root` | repo's `curation/` | Override indices/outputs root |
| `--limit N` | none | Only process first N query items |
| `--content-id ID` | none | Repeat to filter to specific items |
| `--push-to-hub` | false | Push matches.parquet to HF after export |
| `--hf-repo` | auto | Override default HF repo name |

## Output Schema

`matches.parquet` columns (see `schema.py:MATCH_SCHEMA`):

| Column | Type | Description |
|--------|------|-------------|
| `content_id` | str | CleanSlate item ID |
| `content_title` | str | Song/book title |
| `reference_target` | large_string | Full query text |
| `retrieval_method` | str | `infini_gram_mini` or `bm25` |
| `corpus_name` | str | `dolma3_10B` / `dolma3_6T` |
| `doc_id` | str | Dolma document ID |
| `doc_source` | str | Source shard path (e.g. `common_crawl-literature-0018/shard_X.jsonl.zst`) |
| `doc_category` | str | Top-level category dir |
| `unit_id` | str | `{content_id}_span_{start}_{end}` (infini-gram) or `{doc_id}_{segidx}` (bm25) |
| `unit_char_start` | int64 | Span/segment start |
| `unit_char_end` | int64 | Span/segment end |
| `rank` | int64 | Rank within this item (1 = best) |
| `score` | float64 | Span length (infini-gram) or BM25 score |
| `retrieved_text` | large_string | Source document context around match |
| `metadata` | large_string | JSON: span_text, corpus_count, coverage, total_spans, num_docs (infini-gram) or tokenization details (bm25) |

`summary.parquet` aggregates per content_id: `num_hits`, `best_score`, `matched_doc_count`.

## Troubleshooting

- **`AssertionError: fs::exists(index_dir)`** when loading engine — index path doesn't exist. The schema uses a config-derived hash (`{method}/{index_hash}/`); make sure you haven't changed segment params since building.
- **`libcilkrts.so.5` not found** — you're running the pre-compiled `cpp_indexing` binary. Use `cpp_indexing_local` (recompiled without Cilk, see Setup step 2).
- **`ModuleNotFoundError: indexing`** during index build — the backend adds `vendor/infini-gram-mini/src` to `sys.path` on use; check that `VENDOR_ROOT` in `schema.py` resolves to the right dir.
- **`No space left on device`** during SA build — 6T needs ~40TB free. Either use a bigger disk or restrict `--categories`.
- **Empty columns in HF dataset** — stale per-item parquets from a previous schema version. Delete `curation/outputs/{method}_cleanslate_dolma3_{variant}/{hash}/items/` and re-run `search`.
- **Workers hang at "Searching N items..."** — `ProcessPoolExecutor.map` buffers results in chunks; with 128 workers the first batch has to complete before any writes appear. Progress prints every 50 items.
