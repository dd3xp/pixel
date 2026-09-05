#!/bin/bash
# Evaluate a fine-tuned probe at FD-DINOv2 @16px, comparable to v7 baseline=64.8.
# Usage: bash baseline/eval_probe.sh <probe_name> <gpu>   e.g. probe_tv 3
set -eu
NAME=$1; GPU=${2:-3}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=workdir/$NAME/model_latest.pt
[ -f runs_out/derisk_prompts.txt ] || { grep -v '^#' prompts/vocab_distill_v2.txt | sed '/^$/d' | sed 's/^/a pixel art /' > runs_out/derisk_prompts.txt; }
$PY src/v6/sample_e.py --ckpt "$CKPT" \
    --buckets 12,16,20,24,32,48,64 --sizes 16 \
    --prompts runs_out/derisk_prompts.txt --n 8 --seed 0 --out runs_out/${NAME}_eval
$PY - "$NAME" <<'PYEOF'
import glob, random, sys, json
import sys as _s; _s.path.insert(0, "src")
from v6 import fd_dino as F
name = sys.argv[1]
random.seed(0)
REAL = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
random.shuffle(REAL); REAL = REAL[:3000]
gen = sorted(glob.glob(f"runs_out/{name}_eval/s16/*.png"))
fd = F.fd(gen, REAL, 16)
print(f"{name} FD-DINOv2@16 = {fd:.2f}  (n={len(gen)})  vs v7 baseline 64.80", flush=True)
json.dump({"name": name, "fd16": round(fd,2), "n": len(gen), "v7_baseline": 64.80},
          open(f"runs_out/{name}_fd.json","w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
