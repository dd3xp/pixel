#!/bin/bash
# CRSC gates T1 and T2 on GPU2 (zero training; T0 already said STOP on forward-noised data: w* 0.83-1.07).
#  T1: w* on the model's own sampler states (k = 1/5/20 rolled steps, bare and label-guided rollers),
#      16/12, 20/16, 24/16 px. Both T0 and T1 < 1.1 -> drop CRSC as the headline.
#  T2: one-NFE class-embedding extrapolation E[16] + (w-1)(E[16]-E[12]), w = 1.5 / 2, --cfg 1, matched FD@16.
#      Bars: no guidance (cond only) 16.39; best CFG (2 NFE) 12.49; FD <= 10 at 1 NFE would be a free result.
# Launch: tmux new -d -s crsc_t1t2 "bash baseline/run_crsc_t1t2.sh 2>&1 | tee -a logs/crsc_t1t2.log"
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${GPU:-2}
for pair in "16 12" "20 16" "24 16"; do
  set -- $pair
  out=runs_out/crsc_t1_s$1_l$2.json
  [ -f $out ] && continue
  echo "[$(date +%m%d-%H:%M)] T1 size=$1 lower=$2"
  $P src/v6/crsc_t1.py --size $1 --lower $2 --n 2000 --out $out || { echo "CRSC_T1_FAIL size=$1"; touch logs/crsc_t1t2.FAILING; exit 1; }
done
for w in 1.5 2; do
  t=v7h_embx$(echo $w | tr . p)
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  echo "[$(date +%m%d-%H:%M)] T2 emb_extrap w=$w"
  CKPT=workdir/v7h/model_latest.pt EXTRA="--cfg 1 --emb_extrap $w,12" bash baseline/eval_matched_r.sh $t 16 $CUDA_VISIBLE_DEVICES \
    && touch runs_out/${t}_matched_eval/.done || { echo "CRSC_T2_FAIL w=$w"; touch logs/crsc_t1t2.FAILING; exit 1; }
done
echo CRSC_T1T2_ALL_DONE
