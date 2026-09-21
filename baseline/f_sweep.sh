#!/bin/bash
# The other tuned constant: the step at which the projection switches on.  k now has a validation
# split; f still rests on two points (0.5 and 0.7).  Sweep it at k=4 and score each run on both halves
# of the held-out captions, so f is chosen the same way k is.
# Usage: bash baseline/f_sweep.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0.3 0.4 0.6 0.8 1.0; do
  t=v8n_icg2_pal4_f${f/./p}
  if [ ! -f runs_out/${t}_matched_eval/.done ]; then
    CKPT=workdir/v8n/model_latest.pt EXTRA="--guide_mode icg --cfg 2 --pal 4 --pal_from $f --chunk 500" \
      PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
done
# and the same half-split scoring, for every f including the two that already existed
for t in v8n_icg2_pal4 v8n_icg2_pal4_f07 v8n_icg2_pal4_f0p3 v8n_icg2_pal4_f0p4 v8n_icg2_pal4_f0p6 \
         v8n_icg2_pal4_f0p8 v8n_icg2_pal4_f1p0; do
  src=runs_out/${t}_matched_eval/s16
  [ -d "$src" ] || continue
  for h in 0 1; do
    d=runs_out/ksplit/${t}_h${h}
    if [ ! -d "$d" ]; then
      mkdir -p $d
      for g in $src/*.png; do
        n=$(basename $g); i=${n%%_*}; i=${i#0}
        [ -z "$i" ] && i=0
        if [ $((10#$i % 2)) -eq $h ]; then ln -sf ../../../$g $d/$n; fi
      done
    fi
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
      --gen $d --out runs_out/fair_fd16_native_ksplit.json
  done
done
echo FSWEEP_DONE
