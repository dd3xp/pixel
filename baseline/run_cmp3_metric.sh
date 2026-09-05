#!/bin/bash
# 3-way FD-DINOv2 @16px vs real oga_clean: ordinal(v_ord) vs naive(v6f) vs continuous(v7)
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY - <<'PYEOF'
import glob, random, json
import sys; sys.path.insert(0,"src")
from v6 import fd_dino as F
random.seed(0)
REAL = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
random.shuffle(REAL); REAL = REAL[:3000]
print(f"real pool: {len(REAL)}", flush=True)
R = 16
methods = {"ordinal_v_ord": "runs_out/cmp_ord",
           "naive_v6f":     "runs_out/cmp_v6f",
           "continuous_v7": "runs_out/derisk_v7sweep/s16"}
res = {}
mr, sr = F.stats(F.embed(REAL, R))
for name, d in methods.items():
    paths = sorted(glob.glob(f"{d}/*.png"))
    mg, sg = F.stats(F.embed(paths, R))
    fd = F.frechet(mg, sg, mr, sr)
    res[name] = {"fd_dino": round(fd, 2), "n": len(paths)}
    print(f"{name}: FD-DINOv2={fd:.2f} (n={len(paths)})", flush=True)
json.dump(res, open("runs_out/cmp3_fd_dino.json","w"), indent=2)
print("saved runs_out/cmp3_fd_dino.json")
PYEOF
echo CMP3_METRIC_DONE
