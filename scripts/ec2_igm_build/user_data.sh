#!/bin/bash
# EC2 user-data: build ONE IGM shard over a slice of allenai/dolma3_mix-6T
# (5 CC categories). Designed for embarrassingly-parallel fan-out: launch N
# instances with SHARD_INDEX in 0..N-1, each builds shard_{SHARD_INDEX} of
# the final N-shard index.
#
# Assumes i4i.8xlarge (or similar with instance-store NVMe on /dev/nvme1n1)
# running Ubuntu 24.04.
#
# Expected env vars (substituted at launch time via launch_fleet.sh):
#   HF_TOKEN     — HuggingFace token with access to allenai/dolma3_mix-6T
#   S3_BUCKET    — e.g. cleanslate-aws
#   GIT_REF      — branch or commit in akhatua2/CleanSlate (default: main)
#   NUM_SHARDS   — total number of IGM shards in the final index
#   SHARD_INDEX  — which of the N shards this instance builds (0..N-1)
#
# Writes on success:
#   s3://$S3_BUCKET/igm-dolma6T/shard_NNN/{data.fm9,meta.fm9,data_offset,meta_offset}
#   s3://$S3_BUCKET/logs/build-shard_NNN-<instance-id>.log

set -euxo pipefail
exec > >(tee -a /var/log/user-data.log) 2>&1
echo "=== user-data start: $(date -u) ==="

: "${HF_TOKEN:?missing HF_TOKEN}"
: "${S3_BUCKET:=cleanslate-aws}"
: "${GIT_REF:=main}"
: "${NUM_SHARDS:?missing NUM_SHARDS}"
: "${SHARD_INDEX:?missing SHARD_INDEX}"

SHARD_TAG=$(printf "shard_%03d" "$SHARD_INDEX")
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
LOG_KEY="logs/build-${SHARD_TAG}-${INSTANCE_ID}.log"

# -- 1. Mount instance-store NVMe
DEV=$(lsblk -ndo NAME,MOUNTPOINT,TYPE | awk '$3=="disk" && $1 ~ /^nvme[1-9]/ && !$2 {print "/dev/"$1; exit}')
[ -z "$DEV" ] && { lsblk; echo "No unmounted NVMe"; exit 1; }
mkfs.xfs -f "$DEV"
mkdir -p /mnt/nvme && mount "$DEV" /mnt/nvme
chown -R ubuntu:ubuntu /mnt/nvme

# -- 2. OS packages
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git build-essential pkg-config cmake curl ca-certificates \
  awscli zstd unzip xfsprogs

# -- 3. uv
sudo -u ubuntu bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# -- 4. Clone + sync
sudo -u ubuntu -H bash -lc "
  set -euxo pipefail
  source \$HOME/.local/bin/env
  cd /mnt/nvme
  git clone https://github.com/akhatua2/CleanSlate.git
  cd CleanSlate
  git checkout $GIT_REF
  uv sync --extra curation
  .venv/bin/pip install hf_transfer
"

# -- 5. Pull IGM binaries tarball
sudo -u ubuntu -H bash -lc "
  set -euxo pipefail
  cd /mnt/nvme/CleanSlate
  mkdir -p curation/vendor
  aws s3 cp s3://$S3_BUCKET/igm-artifacts/igm-bin-x86_64.tar.gz /tmp/igm-bin.tar.gz
  tar -C curation/vendor -xzf /tmp/igm-bin.tar.gz
  test -x curation/vendor/infini-gram-mini/src/cpp_indexing
"

# -- 6. Download this instance's slice of the corpus
sudo -u ubuntu -H bash -lc "
  set -euxo pipefail
  source \$HOME/.local/bin/env
  cd /mnt/nvme/CleanSlate
  export HF_TOKEN=$HF_TOKEN
  export HF_HUB_ENABLE_HF_TRANSFER=1
  mkdir -p /mnt/nvme/dolma3/raw/6T
  .venv/bin/python scripts/ec2_igm_build/download_shard_slice.py \
    --repo allenai/dolma3_mix-6T \
    --categories common_crawl-entertainment,common_crawl-literature,common_crawl-art_and_design,common_crawl-history_and_geography,common_crawl-religion \
    --num-shards $NUM_SHARDS \
    --shard-index $SHARD_INDEX \
    --local-dir /mnt/nvme/dolma3/raw/6T \
    --workers 8
  find /mnt/nvme/dolma3/raw/6T -name '*.jsonl.zst' | wc -l
"

# -- 7. Build a single FM-index from this slice (= one shard of the final index)
sudo -u ubuntu -H bash -lc "
  set -euxo pipefail
  source \$HOME/.local/bin/env
  cd /mnt/nvme/CleanSlate
  export DOLMA_ROOT=/mnt/nvme/dolma3
  export CURATION_ROOT=/mnt/nvme/curation_work
  mkdir -p \$CURATION_ROOT
  .venv/bin/python -m curation.retrieval.cli build-index \
    --method infini_gram_mini \
    --corpus 6T \
    --dolma-dir \$DOLMA_ROOT \
    --curation-root \$CURATION_ROOT \
    --categories common_crawl-entertainment,common_crawl-literature,common_crawl-art_and_design,common_crawl-history_and_geography,common_crawl-religion \
    --workers 32 \
    --num-index-shards 1 \
    --fm-parallel 1 \
    --index-mem-gb 200
"

# -- 8. Upload the built FM-index files, renamed to shard_NNN/
sudo -u ubuntu -H bash -lc "
  set -euxo pipefail
  INDEX_DIR=\$(find /mnt/nvme/curation_work/indices/6T/infini_gram_mini -maxdepth 3 -name 'data.fm9' | head -1 | xargs dirname)
  echo \"INDEX_DIR=\$INDEX_DIR\"
  test -n \"\$INDEX_DIR\"
  aws s3 cp \"\$INDEX_DIR/data.fm9\"     s3://$S3_BUCKET/igm-dolma6T/$SHARD_TAG/data.fm9
  aws s3 cp \"\$INDEX_DIR/meta.fm9\"     s3://$S3_BUCKET/igm-dolma6T/$SHARD_TAG/meta.fm9
  aws s3 cp \"\$INDEX_DIR/data_offset\"  s3://$S3_BUCKET/igm-dolma6T/$SHARD_TAG/data_offset
  aws s3 cp \"\$INDEX_DIR/meta_offset\"  s3://$S3_BUCKET/igm-dolma6T/$SHARD_TAG/meta_offset
"

# -- 9. Ship log + terminate
aws s3 cp /var/log/user-data.log "s3://$S3_BUCKET/$LOG_KEY" || true
echo "=== user-data end: $(date -u) ==="
shutdown -h now
