#!/bin/bash
# Matched protocol at an arbitrary native resolution R (eval_matched.sh is 16 px only): same 3000 held-out
# captions, n=1, seed 0; fair FD@R against REAL[:3000] -> to_tensor(R) (fd_fair.py --size R), floor computed
# once per R.  No q16 stage.  Usage: CKPT=... EXTRA="..." bash baseline/eval_matched_r.sh <name> <R> <gpu>
set -eu
NAME=$1; R=$2; GPU=${3:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=${CKPT:-workdir/$NAME/model_latest.pt}
[ -f runs_out/heldout3000_prompts.txt ] || { echo "run eval_matched.sh once first (prompts file)"; exit 1; }
$PY src/v6/sample_e.py --ckpt "$CKPT" --buckets 12,16,20,24,32,48,64 --sizes $R \
    --prompts runs_out/heldout3000_prompts.txt --n 1 --seed 0 --out runs_out/${NAME}_matched_eval ${EXTRA:-}
FLOOR=""; grep -q floor_heldout3000 runs_out/fair_fd${R}.json 2>/dev/null || FLOOR=--floor
$PY src/v6/fd_fair.py --size $R $FLOOR --gen runs_out/${NAME}_matched_eval/s$R --out runs_out/fair_fd${R}.json
echo MATCHED_${NAME}_DONE
