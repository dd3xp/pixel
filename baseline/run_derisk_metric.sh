#!/bin/bash
# De-risk phase 2: FD-DINOv2 quality-vs-resolution curve for v7 (feed-forward)
# and SD-piXL (SDS, N=8, flagged), at 12/16/20/24/32.
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"
export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
$PY - <<'PYEOF'
import glob, random, json
import numpy as np
import sys; sys.path.insert(0, "src")
from v6 import fd_dino as F
random.seed(0)

REAL = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
random.shuffle(REAL); REAL = REAL[:3000]
print(f"real pool: {len(REAL)}", flush=True)

rows = {}
for R in [12, 16, 20, 24, 32]:
    v7 = sorted(glob.glob(f"runs_out/derisk_v7sweep/s{R}/*.png"))
    fd_v7 = F.fd(v7, REAL, R)
    # SD-piXL side (only 12/16/20/24 exist, N=8)
    sd = sorted(glob.glob(f"baseline/results/10k_s{R}_p*.png"))
    fd_sd = F.fd(sd, REAL, R) if len(sd) >= 2 else None
    rows[R] = {"v7_fd": round(fd_v7, 2), "v7_n": len(v7),
               "sdpixl_fd": (round(fd_sd, 2) if fd_sd else None), "sdpixl_n": len(sd)}
    print(f"R={R}: v7 FD-DINO={fd_v7:.2f} (n={len(v7)})  SD-piXL FD-DINO={fd_sd if fd_sd is None else round(fd_sd,2)} (n={len(sd)})", flush=True)

json.dump(rows, open("runs_out/derisk_fd_dino.json", "w"), indent=2)
print("saved runs_out/derisk_fd_dino.json")
PYEOF
echo DERISK_METRIC_DONE
