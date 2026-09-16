#!/bin/bash
# 20k-step data-policy pilot (09-16). Control is v7r's own model_step020000.pt: same net, same steps,
# same recaptioned text -- only the target distribution differs, so the comparison isolates the data change.
#   A (quant 0): a sprite may only feed a bucket within 1.5x of its native size, duplicate copies dropped.
#   B (quant 16): A plus median-cut quantisation of every target to <= 16 colours.
# Launch: NEED_MB=20000 setsid nohup bash supervise.sh v8a 6 env QUANT=0 OUT=workdir/v8a bash baseline/run_v8pilot.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-20000}; OUT=${OUT:-workdir/v8a}; QUANT=${QUANT:-0}
TAG=$(basename $OUT)
echo "[$(date +%m%d-%H:%M)] STAGE train $TAG steps=$STEPS quant=$QUANT"
$P src/v6/train_v8.py --steps $STEPS --out $OUT --exclude runs_out/holdout_exclude.txt --csv_suffix _recap \
   --max_down 1.5 --quant $QUANT --dupdrop data/corpus_v8_dupdrop.txt || { echo V8_TRAIN_FAIL; exit 1; }
for w in 1.5 2; do
  t=${TAG}_cfg$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=$OUT/model_latest.pt EXTRA="--cfg $w --chunk 500" PROMPTS=runs_out/heldout3000_prompts_recap.txt \
    bash baseline/eval_matched_r.sh $t 16 ${CUDA_VISIBLE_DEVICES:-6} && touch runs_out/${t}_matched_eval/.done
  $P src/v6/fd_fair.py --size 16 --native --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
done
echo V8_${TAG}_DONE
