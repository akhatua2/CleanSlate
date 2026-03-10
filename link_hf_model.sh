#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model-path>  (e.g. Qwen/Qwen3-8B)" >&2
  exit 1
fi

model="$1"
cache_dir="models--${model/\//--}"
shared="/lfs/skampere3/0/shared_hf_cache/hub/$cache_dir"

if [[ -z "${LOCAL_MACHINE_CACHE:-}" ]]; then
  echo "Error: \$LOCAL_MACHINE_CACHE is not set" >&2
  exit 1
fi

if [[ ! -d "$shared" ]]; then
  echo "Error: $shared does not exist" >&2
  exit 1
fi

dest="$LOCAL_MACHINE_CACHE/hub"
mkdir -p "$dest"

if [[ -d "$dest/$cache_dir" && ! -L "$dest/$cache_dir" ]]; then
  echo "$dest/$cache_dir exists locally, no symlink needed"
  exit 0
fi

if [[ -L "$dest/$cache_dir" ]]; then
  echo "$dest/$cache_dir is already symlinked, skipping"
  exit 0
fi

ln -s "$shared" "$dest/$cache_dir"
echo "Linked: $dest/$cache_dir -> $shared"
