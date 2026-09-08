#!/bin/bash
# q16_eval.sh generalised to any native resolution R: post-hoc 16-colour octree quantisation of
# runs_out/<name>_eval/s<R> -> runs_out/<name>_q16/s<R>, then fair FD@R.  Usage: bash baseline/q16_eval_r.sh <name> <R> <gpu>
set -eu
NAME=$1; R=$2; GPU=${3:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1 HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY - "$NAME" "$R" <<PYEOF
import glob, numpy as np, os, sys
from PIL import Image
name=sys.argv[1]; R=sys.argv[2]; out=f"runs_out/{name}_q16/s{R}"; os.makedirs(out,exist_ok=True); uc=[]
for f in sorted(glob.glob(f"runs_out/{name}_eval/s{R}/*.png")):
    im=Image.open(f).convert("RGBA"); a=np.asarray(im)[...,3]>127
    q=im.quantize(colors=16,method=Image.Quantize.FASTOCTREE).convert("RGBA")
    arr=np.asarray(q).copy(); arr[~a]=0; arr[a,3]=255
    Image.fromarray(arr,"RGBA").save(os.path.join(out,os.path.basename(f)))
    uc.append(len(np.unique(arr[a].reshape(-1,4),axis=0)) if a.any() else 0)
print(name,f"q16@{R} unique colours mean",round(float(np.mean(uc)),1),flush=True)
PYEOF
$PY src/v6/fd_fair.py --size $R --gen runs_out/${NAME}_q16/s$R --out runs_out/fair_fd${R}.json
echo Q16_${NAME}_DONE
