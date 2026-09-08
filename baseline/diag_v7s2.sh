#!/bin/bash
# GPU3 batch after V7S_DONE (paper_outline items 2, 4):
#  (4) clean second model v7s: seed 1 of the four 16 px rows; the four rows at 20 and 24 px (seed 0)  -> Table c / b on a second model
#  (2) v7h seed 2 at 20/24 px (four rows) and 12 px (bare, autog)                                    -> 3-seed mean +- sd at every R
# Launch: setsid nohup bash supervise.sh diag_v7s2 3 bash baseline/diag_v7s2.sh 3 >/dev/null 2>&1 &   (from ssh, < /dev/null)
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
S=workdir/v7s/model_latest.pt;  SW=workdir/v7s/model_step010000.pt
H=workdir/v7h/model_latest.pt;  HW=workdir/v7h/model_step010000.pt
run16() {  # <ckpt> <tag> <extra sample_e args>
  local ck=$1 tag=$2; shift 2
  [ -f runs_out/${tag}_matched_q16/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$ck EXTRA="$*" bash baseline/eval_matched.sh $tag $GPU && touch runs_out/${tag}_matched_q16/.done
}
runR() {  # <ckpt> <R> <tag> <extra sample_e args>
  local ck=$1 R=$2 tag=$3; shift 3
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$ck EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
# (4) v7s seed 1 @16
run16 $S v7s_cfg4_seed1           --cfg 4 --seed 1
run16 $S v7s_gbk12_w2_seed1       --cfg 2 --guide_mode bucket:12 --seed 1
run16 $S v7s_autog10k_w1p5_seed1  --cfg 1.5 --guide_ckpt $SW --seed 1
run16 $S v7s_gbk12s10k_w1p5_seed1 --cfg 1.5 --guide_mode bucket:12 --guide_ckpt $SW --seed 1
# (4) v7s @20 / @24, seed 0
for R in 20 24; do
  runR $S $R v7s_r${R}_cfg4           --cfg 4
  runR $S $R v7s_r${R}_bk16_w2        --cfg 2 --guide_mode bucket:16
  runR $S $R v7s_r${R}_autog10k_w1p5  --cfg 1.5 --guide_ckpt $SW
  runR $S $R v7s_r${R}_bk16s10k_w1p5  --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $SW
done
# (2) v7h seed 2 @20 / @24 / @12
for R in 20 24; do
  runR $H $R v7h_r${R}_cfg4_seed2           --cfg 4 --seed 2
  runR $H $R v7h_r${R}_bk16_w2_seed2        --cfg 2 --guide_mode bucket:16 --seed 2
  runR $H $R v7h_r${R}_autog10k_w1p5_seed2  --cfg 1.5 --guide_ckpt $HW --seed 2
  runR $H $R v7h_r${R}_bk16s10k_w1p5_seed2  --cfg 1.5 --guide_mode bucket:16 --guide_ckpt $HW --seed 2
done
runR $H 12 v7h_r12_cfg4_seed2          --cfg 4 --seed 2
runR $H 12 v7h_r12_autog10k_w1p5_seed2 --cfg 1.5 --guide_ckpt $HW --seed 2
echo "[$(date +%m%d-%H:%M)] fd_decomp v7s seed1"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7s_*seed1_matched_eval/s16)
echo DIAG_V7S2_DONE
