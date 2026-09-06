#!/bin/bash
# Two-stage (structure generator -> colouriser) FAIR FD@16 on the standard 413 prompts x8.
# Usage: bash baseline/eval_twostage.sh <sgen_name> <gpu> [color_name=probe_struct]
set -eu
NAME=$1; GPU=${2:-3}; COLOR=${3:-probe_struct}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/sample_twostage.py --sgen workdir/$NAME/model_latest.pt --color workdir/$COLOR/model_latest.pt \
    --prompts runs_out/derisk_prompts.txt --n 8 --size 16 --out runs_out/${NAME}_eval
$PY src/v6/fd_fair.py --size 16 --gen runs_out/${NAME}_eval/s16 --out runs_out/fair_fd16.json
$PY - "$NAME" <<'PYEOF'
import json, sys
name = sys.argv[1]
fd = json.load(open("runs_out/fair_fd16.json"))[f"runs_out/{name}_eval/s16"]
print(f"{name} TWO-STAGE FAIR FD-DINOv2@16 = {fd:.2f}  vs v7 53.21 / probe_tv 42.82 / floor 3.45", flush=True)
json.dump({"name": name, "fair_fd16": fd, "v7_fair": 53.21, "tv_fair": 42.82, "floor": 3.45, "two_stage": True},
          open(f"runs_out/{name}_fd.json", "w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
