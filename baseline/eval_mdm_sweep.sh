#!/bin/bash
# Sweep CFG weights for MDM to find the best setting before the matched eval.
# Uses the derisk prompts (smaller set) for speed. Usage: bash baseline/eval_mdm_sweep.sh <gpu>
set -eu
GPU=${1:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=${CKPT:-workdir/probe_mdm/model_latest.pt}

[ -f runs_out/derisk_prompts.txt ] || { grep -v '^#' prompts/vocab_distill_v2.txt | sed '/^$/d' | sed 's/^/a pixel art /' > runs_out/derisk_prompts.txt; }

for CFG in 1.0 1.5 2.0 3.0 4.0; do
  echo "--- CFG=$CFG ---"
  TAG=probe_mdm_cfg${CFG}
  $PY src/v6/sample_mdm.py --ckpt "$CKPT" --sizes 16 --prompts runs_out/derisk_prompts.txt \
      --n 8 --seed 0 --cfg "$CFG" --steps 64 --out runs_out/${TAG}_eval
  $PY src/v6/fd_fair.py --size 16 --gen runs_out/${TAG}_eval/s16 --out runs_out/fair_fd16.json
  FD=$($PY -c "import json; print(json.load(open('runs_out/fair_fd16.json'))[f'runs_out/${TAG}_eval/s16'])")
  echo "cfg=$CFG -> FD=$FD"
done
echo MDM_SWEEP_DONE
