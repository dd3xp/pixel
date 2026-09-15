#!/bin/bash
# Is 1b + ICG = 6.25 (original 1b model) reproducible on an independent re-training (probe_gft_v7r_b, 10k)?
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
CK=workdir/probe_gft_v7r_b/model_step010000.pt
for s in 0 1 2; do
  t=v7r_gftb_w1p25_icg1p5; [ $s != 0 ] && t=${t}_seed$s
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  for try in 1 2 3; do
    CKPT=$CK EXTRA="--gft_beta 0.8 --cfg 1.5 --guide_mode icg --seed $s --chunk 500" bash baseline/eval_matched_r.sh $t 16 $GPU \
      && { touch runs_out/${t}_matched_eval/.done; break; }
    sleep 300
  done
done
for s in 1 2; do
  t=v7r_gftb_s010000_w1p5_seed$s
  [ -f runs_out/${t}_matched_eval/.done ] && continue
  CKPT=$CK EXTRA="--cfg 1 --gft_beta 0.666667 --seed $s --chunk 500" bash baseline/eval_matched_r.sh $t 16 $GPU && touch runs_out/${t}_matched_eval/.done
done
echo STACKB_DONE
