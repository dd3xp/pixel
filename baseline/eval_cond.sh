#!/bin/bash
# Oracle FAIR FD-DINOv2@16 for a train_cond.py probe (side-condition from held-out real
# sprites disjoint from the reference). Usage: bash baseline/eval_cond.sh <name> <gpu>
# NOTE: 413 held-out sources x8 near-duplicates -> small-n floor is 13.82, not 3.45.
set -eu
NAME=$1; GPU=${2:-3}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/sample_cond.py --ckpt workdir/$NAME/model_latest.pt --n_src 413 --n 8 --size 16 --out runs_out/${NAME}_eval
$PY src/v6/fd_fair.py --size 16 --gen runs_out/${NAME}_eval/s16 --out runs_out/fair_fd16.json
$PY - "$NAME" <<'PYEOF'
import json, sys
name = sys.argv[1]
fd = json.load(open("runs_out/fair_fd16.json"))[f"runs_out/{name}_eval/s16"]
print(f"{name} ORACLE FAIR FD-DINOv2@16 = {fd:.2f}  vs v7 53.21 / 413-src floor 13.82", flush=True)
json.dump({"name": name, "fair_fd16": fd, "v7_fair": 53.21, "floor_413src": 13.82, "oracle": True},
          open(f"runs_out/{name}_fd.json", "w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
