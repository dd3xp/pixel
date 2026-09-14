#!/bin/bash
# Win 16 px clearly (user 09-15: 16 px is the main battlefield; no change of paper line).
# Now: 1b (1 NFE) 7.23 +- 0.45 vs ICG (2 NFE) 7.46 +- 0.37 -- a tie. Target <= ~6.8.
# Stage A (sampling only, v7r 1b model): inference weight sweep, and 1b + ICG stacking (the internalised guidance acts
#   on the resolution axis, ICG on the text axis; 1b + CFG hurt, ICG is a different reference).
# Stage B: continue 1b training 10k -> 20k steps (resume from its ckpt; 10k weights kept), eval w1.5 seeds 0-2.
# Launch: NEED_MB=10000 setsid nohup bash supervise.sh improve16 2 bash baseline/run_improve16.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
OUT=workdir/probe_gft_v7r
# freeze the 10k weights first: stage B overwrites model_latest.pt while other evals may still read the 10k model
[ -f $OUT/model_step010000_kept.pt ] || cp $OUT/model_latest.pt $OUT/model_step010000_kept.pt
CK=$OUT/model_step010000_kept.pt
ev() {  # <tag> <ckpt> <args...>  (retries: the card is shared)
  local tag=$1 ck=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$ck EXTRA="$* --chunk 500" bash baseline/eval_matched_r.sh $tag 16 $GPU && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    echo "IMPROVE16_RETRY $tag $try"; sleep 300
  done
}
# Stage A
ev v7r_gft_w1p35 $CK --cfg 1 --gft_beta 0.740741
ev v7r_gft_w1p75 $CK --cfg 1 --gft_beta 0.571429
ev v7r_gft_w1p5_icg1p25 $CK --gft_beta 0.666667 --cfg 1.25 --guide_mode icg
ev v7r_gft_w1p5_icg1p5 $CK --gft_beta 0.666667 --cfg 1.5 --guide_mode icg
ev v7r_gft_w1p25_icg1p5 $CK --gft_beta 0.8 --cfg 1.5 --guide_mode icg
echo IMPROVE16_A_DONE
# Stage B: 20k steps (needs ~34 GB: waits for memory inside the training call via retries)
until grep -q GFT_V7R_DONE logs/gft_v7r.log && [ -z "$(ps -eo cmd | grep '[r]un_gft_v7r')" ]; do sleep 120; done
if [ ! -f $OUT/.trained20k ]; then
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)" -ge 34000 ]; do sleep 120; done
  $P src/v6/train_gft_res.py --steps 20000 --out $OUT --ref snapshot --init workdir/v7r/model_latest.pt \
     --snap workdir/v7r_snap10k/model_latest.pt --csv_suffix _recap >> logs/improve16_train20k.log 2>&1 \
     && touch $OUT/.trained20k
fi
cp $OUT/model_latest.pt $OUT/model_step020000_kept.pt
for s in 0 1 2; do
  t=v7r_gft20k_w1p5; [ $s != 0 ] && t=${t}_seed$s
  ev $t $OUT/model_step020000_kept.pt --cfg 1 --gft_beta 0.666667 --seed $s
done
echo IMPROVE16_DONE
