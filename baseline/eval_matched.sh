#!/bin/bash
# Prompt-distribution control: sample with the (auto)captions of the 3000 HELD-OUT real sprites
# (fd_fair.real_split, disjoint from the reference), n=1 each, instead of the 413 vocab prompts x8.
# Isolates how much of the FD gap is prompt-distribution mismatch.  Usage: bash baseline/eval_matched.sh <probe> <gpu>
set -eu
NAME=$1; GPU=${2:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=${CKPT:-workdir/$NAME/model_latest.pt}
[ -f runs_out/heldout3000_prompts.txt ] || $PY - <<'PYEOF'
import csv, os, sys
sys.path.insert(0, "src/v6"); sys.path.insert(0, "src")
from fd_fair import real_split
cap = {r["path"]: r["text"] for r in csv.DictReader(open("data/oga_captions.csv", encoding="utf-8"))}
_, held = real_split()
lines = [cap.get(os.path.basename(p), "a pixel art sprite").strip() or "a pixel art sprite" for p in held[:3000]]
open("runs_out/heldout3000_prompts.txt", "w", encoding="utf-8").write("\n".join(l.replace("\n", " ") for l in lines))
print("wrote", len(lines), "prompts")
PYEOF
$PY src/v6/sample_e.py --ckpt "$CKPT" --buckets 12,16,20,24,32,48,64 --sizes 16 \
    --prompts runs_out/heldout3000_prompts.txt --n 1 --seed 0 --out runs_out/${NAME}_matched_eval ${EXTRA:-}
$PY src/v6/fd_fair.py --size 16 --gen runs_out/${NAME}_matched_eval/s16 --out runs_out/fair_fd16.json
bash baseline/q16_eval.sh ${NAME}_matched $GPU
echo MATCHED_${NAME}_DONE
