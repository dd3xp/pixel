#!/bin/bash
# probe_vpred (cycle 9, architecture loop, diagnostic): this project has never varied the training objective -- every
# model since v6 is eps-prediction.  At 12-32 px in pixel space, eps-prediction weights the loss towards the high-noise
# end where the target is nearly the noise itself, and the literature on pixel-space diffusion argues this is what
# breaks low-resolution models.  probe_vpred is v7h with the single change eps -> v-prediction: same data, exclusions,
# width, steps, lr, EMA and schedule, so the comparison is paired.
# It has no novelty on its own; it is run because every later architecture inherits the objective, and it is cheap.
# Bar: v7h best-CFG 12.49, current best (zero-training composed guidance) 7.53, floor 3.45.
# Sampling MUST pass --pred v_prediction (sample_e now takes that flag).
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh probe_vpred 3 bash baseline/run_probe_vpred.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-80000}
OUT=${OUT:-workdir/probe_vpred}
EX=runs_out/holdout_exclude.txt
mkdir -p runs_out logs
echo "[$(date +%m%d-%H:%M)] STAGE train probe_vpred steps=$STEPS (v7h recipe, v-prediction)"
$P src/v6/train_vpred.py --steps $STEPS --out $OUT --exclude $EX --snap_every 5000 || { echo PROBE_VPRED_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE matched eval @16 (CFG sweep, since the optimal weight may move with the objective)"
for w in 1.5 2 3 4; do
  t=vpred_cfg$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=$OUT/model_latest.pt EXTRA="--cfg $w --pred v_prediction" bash baseline/eval_matched_r.sh $t 16 ${CUDA_VISIBLE_DEVICES:-3} && touch runs_out/${t}_matched_eval/.done
done
echo PROBE_VPRED_DONE
