#!/bin/bash
# De-risk phase 2b (fair comparison): same 8 baseline prompts, matched, per-image
# DINOv2 real-manifold proximity for v7 vs SD-piXL at each resolution. Valid at
# small N; the reference (real@R) is identical within a resolution, so the
# v7-vs-SDpiXL GAP at each R is a fair signal. Question: does the gap widen as
# resolution shrinks (i.e. does SDS collapse faster)?
set -eu
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel
cd "$ROOT"
export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
# 1) v7 on the exact 8 baseline prompts, 8 samples each, at 12/16/20/24
$PY src/v6/sample_e.py --ckpt workdir/v7_lowres/model_latest.pt \
    --buckets 12,16,20,24,32,48,64 --sizes 12 16 20 24 \
    --prompts baseline/prompts8.txt --n 8 --seed 0 --out runs_out/derisk_v7_p8
# 2) per-image DINO real-manifold proximity
$PY - <<'PYEOF'
import glob, random, json
import numpy as np, torch
import sys; sys.path.insert(0, "src")
from v6 import fd_dino as F
random.seed(0)

REAL = glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
random.shuffle(REAL); REAL = REAL[:2000]

def proximity(gen_paths, real_emb, R, k=10):
    ge = F.embed(gen_paths, R)
    ge = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
    sims = ge @ real_emb.T                      # cosine (real_emb normalized)
    topk = np.sort(sims, axis=1)[:, -k:].mean(1)  # mean sim to k nearest real
    return topk                                  # per-image realism, higher=better

rows = {}
for R in [12, 16, 20, 24]:
    re = F.embed(REAL, R); re = re / (np.linalg.norm(re, axis=1, keepdims=True) + 1e-8)
    v7 = sorted(glob.glob(f"runs_out/derisk_v7_p8/s{R}/*.png"))
    sd = sorted(glob.glob(f"baseline/results/10k_s{R}_p*.png"))
    pv, ps = proximity(v7, re, R), proximity(sd, re, R)
    rows[R] = {"v7_prox": round(float(pv.mean()), 4), "v7_std": round(float(pv.std()), 4), "v7_n": len(v7),
               "sdpixl_prox": round(float(ps.mean()), 4), "sdpixl_std": round(float(ps.std()), 4), "sdpixl_n": len(sd),
               "gap_v7_minus_sd": round(float(pv.mean() - ps.mean()), 4)}
    print(f"R={R}: v7 realism={pv.mean():.4f}±{pv.std():.3f} (n={len(v7)})  SD-piXL={ps.mean():.4f}±{ps.std():.3f} (n={len(sd)})  gap={pv.mean()-ps.mean():+.4f}", flush=True)

json.dump(rows, open("runs_out/derisk_paired.json", "w"), indent=2)
print("saved runs_out/derisk_paired.json")
PYEOF
echo DERISK_PAIRED_DONE
