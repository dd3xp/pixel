#!/bin/bash
# Make every compared cell 3 seeds before the paper tables are frozen (09-21).
#   12 px (v7r8): best CFG w1.5 and label bucket:8 w2.5, seeds 1-2 (1b already 3 seeds: 7.55 +- 0.19)
#   24 px (v7r): composed (snapshot + bucket:16, w1.5) seed 2 (seeds 0/1: 50.26 / 47.95)
#   16 px (v7r): CDG w1.5 seeds 1-2 (seed 0: 10.36), the closest external guidance method
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1
GPU=${CUDA_VISIBLE_DEVICES:-2}
ev() {  # <tag> <R> <ckpt> <args>
  local tag=$1 R=$2 ck=$3; shift 3
  [ -f runs_out/${tag}_matched_eval/.done ] && return 0
  for try in 1 2 3; do
    CKPT=$ck EXTRA="$* --chunk 500" bash baseline/eval_matched_r.sh $tag $R $GPU && { touch runs_out/${tag}_matched_eval/.done; return 0; }
    sleep 300
  done
}
B8=12,16,20,24,32,48,64,8
for s in 1 2; do
  ev v7r8_r12_cfg1p5_seed$s 12 workdir/v7r8/model_latest.pt --buckets $B8 --cfg 1.5 --seed $s
  ev v7r8_r12_bk8_w2p5_seed$s 12 workdir/v7r8/model_latest.pt --buckets $B8 --cfg 2.5 --guide_mode bucket:8 --seed $s
  ev v7r_cdg_w1p5_seed$s 16 workdir/v7r/model_latest.pt --cfg 1.5 --guide_mode cdg --seed $s
done
ev v7r_r24_stk16_w1p5_seed2 24 workdir/v7r/model_latest.pt --cfg 1.5 --guide_ckpt workdir/v7r_snap10k/model_latest.pt --guide_mode bucket:16 --seed 2
echo FILL_SEEDS_DONE
