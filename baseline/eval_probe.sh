#!/bin/bash
# Evaluate a fine-tuned probe at FAIR FD-DINOv2 @16px (pipeline-matched reference, see
# src/v6/fd_fair.py). Baselines (2026-09-06): real floor 3.45, v7 53.21, probe_tv 42.82.
# Usage: bash baseline/eval_probe.sh <probe_name> <gpu>   e.g. probe_tv 3
set -eu
NAME=$1; GPU=${2:-3}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=${CKPT:-workdir/$NAME/model_latest.pt}   # env override: CKPT=..., EXTRA='--cfg 2 --sampler ddim' for sampler diagnostics
EXTRA=${EXTRA:-}
[ -f runs_out/derisk_prompts.txt ] || { grep -v '^#' prompts/vocab_distill_v2.txt | sed '/^$/d' | sed 's/^/a pixel art /' > runs_out/derisk_prompts.txt; }
$PY src/v6/sample_e.py --ckpt "$CKPT" \
    --buckets 12,16,20,24,32,48,64 --sizes 16 \
    --prompts runs_out/derisk_prompts.txt --n 8 --seed 0 --out runs_out/${NAME}_eval $EXTRA
$PY src/v6/fd_fair.py --size 16 --gen runs_out/${NAME}_eval/s16 --out runs_out/fair_fd16.json
$PY - "$NAME" <<'PYEOF'
import json, sys
name = sys.argv[1]
fd = json.load(open("runs_out/fair_fd16.json"))[f"runs_out/{name}_eval/s16"]
print(f"{name} FAIR FD-DINOv2@16 = {fd:.2f}  vs v7 53.21 / probe_tv 42.82 / real floor 3.45", flush=True)
json.dump({"name": name, "fair_fd16": fd, "v7_fair": 53.21, "tv_fair": 42.82, "floor": 3.45},
          open(f"runs_out/{name}_fd.json", "w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
