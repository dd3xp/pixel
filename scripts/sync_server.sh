#!/usr/bin/env bash
# Sync repo to emnlp server (excluding outputs/git). Run from repo root.
set -e
rsync -av --delete \
  --exclude '.git' --exclude 'workdir' --exclude '__pycache__' \
  ./ emnlp:/mnt/data/kw/RoundSquisheen/texture/pixel/
echo "Synced. On server:"
echo "  cd /mnt/data/kw/RoundSquisheen/texture/pixel"
echo "  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 \\"
echo "    /mnt/data/kw/anaconda3/envs/SD-piXL/bin/python scripts/run.py -c configs/petal_16_db32.yaml"
