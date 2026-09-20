#!/bin/bash
# Full-length run on the repaired target distribution (09-16).  Policy comes from the 20k pilots:
#   v8a (<=1.5x downscale + exact-duplicate drop) beat the v7r control on BOTH protocols at equal steps
#   (24.67 -> 20.31 default, 209.60 -> 151.85 native), so it is the one taken to full length.
#   Quantised targets (v8b) win the native protocol but lose the default one, so quantisation stays a
#   post-process / ablation rather than the main recipe.
# DROP selects the exclusion list: dropall also collapses near-duplicate animation frames (29% of the old
# corpus), which v8c tests at 20k in parallel.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
STEPS=${STEPS:-80000}; OUT=${OUT:-workdir/v8full}; DROP=${DROP:-data/corpus_v8_dropall.txt}; QUANT=${QUANT:-0}
# EXTRA_SRC adds one more source as "<img_dir>,<captions_csv>,<repeat>".  Used for the 09-17 corpus
# (data/oga3_clean + data/oga3_captions_recap.csv: 13,987 natively small sprites, captioned with
# gemini-3.8-flash), i.e. the half of the data problem that the bucket-policy fixes cannot reach.
EXTRA_SRC=${EXTRA_SRC:-}
# CSV_SUFFIX picks the caption set: _recap (plain) or _q (craft-quality tag prepended, 09-18)
EXTRA_SRC2=${EXTRA_SRC2:-}      # a second added source, e.g. the CC0 Kenney corpus (09-20)
EXTRA_SRC3=${EXTRA_SRC3:-}      # a third, e.g. the CC-BY-3.0 bundle (09-20)
EX_ARG=""; [ -n "$EXTRA_SRC" ] && EX_ARG="--extra $EXTRA_SRC"
[ -n "$EXTRA_SRC2" ] && EX_ARG="$EX_ARG --extra $EXTRA_SRC2"
[ -n "$EXTRA_SRC3" ] && EX_ARG="$EX_ARG --extra $EXTRA_SRC3"
TAG=$(basename $OUT)
[ -f $OUT/.trained ] || {
  echo "[$(date +%m%d-%H:%M)] STAGE train $TAG steps=$STEPS drop=$DROP quant=$QUANT"
  $P src/v6/train_v8.py --steps $STEPS --out $OUT --exclude runs_out/holdout_exclude.txt --csv_suffix ${CSV_SUFFIX:-_recap} \
     --max_down 1.5 --quant $QUANT --dupdrop $DROP --snap_every 20000 $EX_ARG ${ARCH_ARG:-} || { echo V8FULL_TRAIN_FAIL; exit 1; }
  touch $OUT/.trained
}
for w in 1.5 2; do
  t=${TAG}_cfg$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=$OUT/model_latest.pt EXTRA="--cfg $w --chunk 500" PROMPTS=runs_out/heldout3000_prompts_recap.txt \
    bash baseline/eval_matched_r.sh $t 16 ${CUDA_VISIBLE_DEVICES:-6} && touch runs_out/${t}_matched_eval/.done
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6} $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
done
echo V8FULL_${TAG}_DONE
