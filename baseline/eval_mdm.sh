#!/bin/bash
# Evaluate MDM (A1 masked discrete diffusion) probe at FAIR FD-DINOv2 @16px.
# Matched protocol: 3000 held-out prompts, n=1, seed 0.
# Usage: bash baseline/eval_mdm.sh <gpu>   e.g.  bash baseline/eval_mdm.sh 2
# Override: CKPT=... CFG=... STEPS=... TEMP=... bash baseline/eval_mdm.sh <gpu>
set -eu
NAME=probe_mdm
GPU=${1:-2}
ROOT=/mnt/data/kw/RoundSquisheen/pixel/pixel; cd "$ROOT"; export PYTHONNOUSERSITE=1
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=$GPU
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
CKPT=${CKPT:-workdir/$NAME/model_latest.pt}
CFG=${CFG:-2.0}
STEPS=${STEPS:-64}
TEMP=${TEMP:-1.0}

[ -f runs_out/heldout3000_prompts.txt ] || { echo "need runs_out/heldout3000_prompts.txt (run eval_matched.sh once first)"; exit 1; }

echo "=== eval MDM: cfg=$CFG steps=$STEPS temp=$TEMP ckpt=$CKPT ===" >&2
$PY src/v6/sample_mdm.py --ckpt "$CKPT" \
    --sizes 16 --prompts runs_out/heldout3000_prompts.txt \
    --n 1 --seed 0 --cfg "$CFG" --steps "$STEPS" --temperature "$TEMP" \
    --out runs_out/${NAME}_matched_eval

$PY src/v6/fd_fair.py --size 16 --gen runs_out/${NAME}_matched_eval/s16 --out runs_out/fair_fd16.json

$PY - "$NAME" "$CFG" "$STEPS" <<'PYEOF'
import json, sys
name, cfg, steps = sys.argv[1], sys.argv[2], sys.argv[3]
fd = json.load(open("runs_out/fair_fd16.json"))[f"runs_out/{name}_matched_eval/s16"]
print(f"{name} (cfg={cfg}, steps={steps}) FAIR FD-DINOv2@16 = {fd:.2f}", flush=True)
print(f"  vs best-CFG 12.49 / composite 7.53 / real floor 3.45", flush=True)
tag = "PASS (<7.5)" if fd < 7.5 else "ZONE (7.5-12)" if fd < 12 else "KILL (>12)"
print(f"  verdict: {tag}", flush=True)
json.dump({"name": name, "cfg": float(cfg), "steps": int(steps),
           "fair_fd16": fd, "verdict": tag},
          open(f"runs_out/{name}_fd.json", "w"), indent=2)
PYEOF
echo EVAL_${NAME}_DONE
