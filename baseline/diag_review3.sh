#!/bin/bash
# Best-CFG baseline, part 2 (GPU2 after diag_review1a): extend the CFG curve upward at 20/24/32 px (w=2.5 @20, w=3 @20/24/32),
# then run seeds 1/2 of the best-w CFG row at 20 and 24 px (best chosen from the FAIR FD lines already in the logs, so the
# 3-seed "best CFG" rows of Table (b) are honest).
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review3 2 bash baseline/diag_review3.sh 2 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt
runR() {  # <tag> <R> <extra sample_e args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
runR v7h_r20_cfg2p5 20 --cfg 2.5
runR v7h_r20_cfg3   20 --cfg 3
runR v7h_r24_cfg3   24 --cfg 3
runR v7h_r32_cfg3   32 --cfg 3
bestw() {  # <R> -> best w among the seed-0 CFG rows (w=4 row is v7h_r${R}_matched_eval)
  { grep -hE "FAIR FD@$1 runs_out/v7h_r$1_cfg[0-9p]+_matched_eval/" logs/*.log | sed -E 's#.*_cfg([0-9p]+)_matched_eval.* = ([0-9.]+)#\2 \1#';
    grep -hE "FAIR FD@$1 runs_out/v7h_r$1_matched_eval/" logs/*.log | head -n 1 | sed -E 's#.* = ([0-9.]+)#\1 4#'; } | sort -n | head -n 1 | awk '{print $2}'
}
for R in 20 24; do
  w=$(bestw $R); echo "[$(date +%m%d-%H:%M)] best CFG w at $R px = $w"
  [ "$w" = 4 ] && continue   # w=4 seeds already exist
  for s in 1 2; do runR v7h_r${R}_cfg${w}_seed${s} $R --cfg ${w/p/.} --seed $s; done
done
echo DIAG_REVIEW3_DONE
