#!/bin/bash
# v7r = v7h retrained on the RECAPTIONED corpus. Everything else (data, exclusions, width, steps, lr, EMA,
# schedule) is v7h exactly, so v7r vs v7h isolates the caption change.
# Why this must exist: every number the project quotes -- best-CFG 12.49, composed guidance 7.53, the external
# comparison at n=200 -- was produced on BLIP captions where 15.4% of rows were literally "a pixel art sprite"
# and only 23.5% were distinct. After recaptioning (98.6% distinct, median 12 words) the old numbers are no
# longer comparable, so the baseline has to be re-established before any new architecture is judged against it.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh v7r 3 bash baseline/run_v7r.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-80000}
OUT=${OUT:-workdir/v7r}
EX=runs_out/holdout_exclude.txt
echo "[$(date +%m%d-%H:%M)] STAGE train v7r on recaptioned corpus, steps=$STEPS"
$P src/v6/train_v7.py --steps $STEPS --out $OUT --exclude $EX --snap_every 20000 \
   --csv_suffix _recap || { echo V7R_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE CFG sweep (the optimal weight may move with better text conditioning)"
for w in 1.5 2 3 4; do
  t=v7r_cfg$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=$OUT/model_latest.pt EXTRA="--cfg $w" bash baseline/eval_matched_r.sh $t 16 ${CUDA_VISIBLE_DEVICES:-3} \
    && touch runs_out/${t}_matched_eval/.done
done
echo V7R_DONE
