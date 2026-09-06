#!/bin/bash
# Oracle FD-DINOv2@16 for a train_cond.py probe. Usage: bash baseline/eval_cond.sh <name> <gpu>
set -eu
NAME=$1; GPU=${2:-3}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY src/v6/sample_cond.py --ckpt workdir/$NAME/model_latest.pt --n_src 413 --n 8 --size 16 --out runs_out/${NAME}_eval
$PY - "$NAME" <<'PYEOF'
import glob, random, sys, json
sys.path.insert(0, "src")
from v6 import fd_dino as F
name = sys.argv[1]
random.seed(0)
REAL = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
random.shuffle(REAL); REAL = REAL[:3000]
gen = sorted(glob.glob(f"runs_out/{name}_eval/s16/*.png"))
fd = F.fd(gen, REAL, 16)
print(f"{name} ORACLE FD-DINOv2@16 = {fd:.2f}  (n={len(gen)})  vs v7 baseline 64.80", flush=True)
json.dump({"name": name, "fd16": round(fd,2), "n": len(gen), "v7_baseline": 64.80, "oracle": True},
          open(f"runs_out/{name}_fd.json","w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
