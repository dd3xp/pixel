#!/bin/bash
# Control: sample a model at SRC px (default 32), box-downsample with the training to_tensor
# pipeline to 16px, score at fair FD@16.  Tests whether "generate coarse-from-fine" beats native 16px.
# Usage: bash baseline/ctrl_down16.sh <probe_name> <gpu> [SRC]
set -eu
NAME=$1; GPU=${2:-3}; SRC=${3:-32}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
OUT=runs_out/${NAME}_s${SRC}to16
[ -d $OUT/s${SRC} ] || $PY src/v6/sample_e.py --ckpt workdir/$NAME/model_latest.pt \
    --buckets 12,16,20,24,32,48,64 --sizes $SRC \
    --prompts runs_out/derisk_prompts.txt --n 8 --seed 0 --out $OUT
$PY - "$OUT" "$SRC" <<'PYEOF'
import glob, os, sys
sys.path.insert(0, "src/v6")
import numpy as np
from PIL import Image
from train_v7 import to_tensor
out, src = sys.argv[1], sys.argv[2]
dst = f"{out}/s16"; os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(f"{out}/s{src}/*.png")):
    im = Image.open(f).convert("RGBA")
    bb = im.getbbox()
    if bb: im = im.crop(bb)  # to_tensor works on the sprite's own bbox (like dataset sources)
    x = to_tensor(im, 16)
    Image.fromarray(((x + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy(), "RGBA").save(f"{dst}/{os.path.basename(f)}")
print("downsampled", len(glob.glob(f"{dst}/*.png")))
PYEOF
$PY src/v6/fd_fair.py --size 16 --gen $OUT/s16 --out runs_out/fair_fd16.json
echo CTRL_${NAME}_${SRC}_DONE
