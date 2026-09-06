#!/bin/bash
# v7h = v7 recipe retrained with a CLEAN evaluation hold-out.
# v7 was initialised from v6e10 (trained on all of oga_clean) and itself saw every oga sprite,
# so the "matched" protocol (captions of held[:3000]) was memorisation-contaminated
# (8.4% near-exact copies).  v7h: random init, drops ref[:3000] + held[:3000] from every
# source, keeps EMA snapshots every 5k steps (autoguidance bad model = early snapshot).
# All later probes fine-tune from v7h and are judged against v7h under eval_matched.sh.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-80000}
OUT=${OUT:-workdir/v7h}
mkdir -p runs_out logs
EX=runs_out/holdout_exclude.txt
[ -s $EX ] || $P - <<'EOF'
import sys; sys.path.insert(0, "src/v6")
from fd_fair import real_split
ref, held = real_split()
paths = sorted({p.replace("\\", "/") for p in ref} | {p.replace("\\", "/") for p in held[:3000]})
open("runs_out/holdout_exclude.txt", "w").write("\n".join(paths) + "\n")
print("exclude", len(paths))
EOF
echo "[$(date +%m%d-%H:%M)] STAGE train v7h steps=$STEPS"
$P src/v6/train_v7.py --steps $STEPS --out $OUT --exclude $EX --snap_every 5000 || { echo V7H_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE matched eval"
NAME=v7h CKPT=$OUT/model_latest.pt bash baseline/eval_matched.sh
$P src/v6/fd_decomp.py --size 16 --gen runs_out/v7h_matched_eval/s16
echo V7H_DONE
