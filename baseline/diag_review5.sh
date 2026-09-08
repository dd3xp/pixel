#!/bin/bash
# Fill the [TBD] cells the best-CFG rewrite left open (paper_assets/draft_*.md):
#  (a) CFG w=2.5 @16 (the only grid hole at 16 px); (b) 50-step best-CFG @16; (c) q16 of CFG w=1.5 @16;
#  (d) CLIP / R@1 of the best-CFG samples at 12/20/24/32 px; (e) Inception FID/KID of the best-CFG samples at every R.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review5 3 bash baseline/diag_review5.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt
runR() {  # <tag> <R> <extra sample_e args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
runR v7h_cfg2p5      16 --cfg 2.5
runR v7h_st50_cfg1p5 16 --cfg 1.5 --steps 50
echo "[$(date +%m%d-%H:%M)] q16 of CFG w=1.5"
[ -f runs_out/v7h_cfg1p5_matched_q16/.done ] || { bash baseline/q16_eval.sh v7h_cfg1p5_matched $GPU && touch runs_out/v7h_cfg1p5_matched_q16/.done; }
echo "[$(date +%m%d-%H:%M)] CLIP of best-CFG rows at 12/20/24/32 (and w=4 rows for reference)"
M=runs_out; E=_matched_eval
$P src/v6/clip_score.py --size 12 --out runs_out/clip_scores.json --dirs r12_cfg2=$M/v7h_r12_cfg2$E/s12 r12_cfg4=$M/v7h_r12_cfg4$E/s12 r12_autog=$M/v7h_r12_autog10k_w1p5$E/s12
$P src/v6/clip_score.py --size 20 --out runs_out/clip_scores.json --dirs r20_cfg2p5=$M/v7h_r20_cfg2p5$E/s20 r20_cfg4=$M/v7h_r20_cfg4$E/s20 r20_bk16=$M/v7h_r20_bk16_w2$E/s20 r20_composed=$M/v7h_r20_bk16s10k_w1p5$E/s20
$P src/v6/clip_score.py --size 24 --out runs_out/clip_scores.json --dirs r24_cfg2=$M/v7h_r24_cfg2$E/s24 r24_cfg4=$M/v7h_r24_cfg4$E/s24 r24_bk16=$M/v7h_r24_bk16_w2$E/s24 r24_composed=$M/v7h_r24_bk16s10k_w1p5$E/s24
$P src/v6/clip_score.py --size 32 --out runs_out/clip_scores.json --dirs r32_cfg2=$M/v7h_r32_cfg2$E/s32 r32_cfg4=$M/v7h_r32_cfg4$E/s32 r32_bk24=$M/v7h_r32_bk24_w2$E/s32 r32_composed=$M/v7h_r32_bk24s10k_w1p5$E/s32
$P src/v6/clip_score.py --size 16 --out runs_out/clip_scores.json --dirs cfg2p5=$M/v7h_cfg2p5$E/s16 st50_cfg1p5=$M/v7h_st50_cfg1p5$E/s16 st50_composed=$M/v7h_st50_composed$E/s16
echo "[$(date +%m%d-%H:%M)] Inception FID/KID of best-CFG samples"
$P src/v6/fid_kid_fair.py --size 16 --out runs_out/incep_scores.json --gen $M/v7h_cfg1p5$E/s16 $M/v7h_cfg2$E/s16 $M/v7h_cfg1$E/s16 $M/v7h_cfg3$E/s16 $M/v7h_pagmid_w2$E/s16
$P src/v6/fid_kid_fair.py --size 12 --out runs_out/incep_scores.json --gen $M/v7h_r12_cfg2$E/s12
$P src/v6/fid_kid_fair.py --size 20 --out runs_out/incep_scores.json --gen $M/v7h_r20_cfg2p5$E/s20
$P src/v6/fid_kid_fair.py --size 24 --out runs_out/incep_scores.json --gen $M/v7h_r24_cfg2$E/s24
$P src/v6/fid_kid_fair.py --size 32 --out runs_out/incep_scores.json --gen $M/v7h_r32_cfg2$E/s32
echo DIAG_REVIEW5_DONE
