#!/bin/bash
# paper_outline items 10 + 8 (GPU2), on SAVED samples, no resampling:
#  (10) CLIP score (text faithfulness) of real + the main rows at 12/16/20/24/32 px, seeds 0 and 1 where they exist
#  (8)  q16 appendix at 20/24 px for the four main rows (seed 0) + the 32 px rows
# Usage: bash baseline/diag_clipq16.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
M=runs_out; E=_matched_eval
clip() { $P src/v6/clip_score.py --size $1 --out runs_out/clip_scores.json --dirs "${@:2}"; }
echo "[$(date +%m%d-%H:%M)] CLIP 16"
clip 16 real=$M/heldout3000_totensor_s16 cfg4_s0=$M/v7h$E/s16 cfg4_s1=$M/v7h_seed1$E/s16 cfg4_s2=$M/v7h_seed2$E/s16 \
  autog_s0=$M/v7h_autog10k_w1p5$E/s16 autog_s1=$M/v7h_autog10k_w1p5_seed1$E/s16 autog_s2=$M/v7h_autog10k_w1p5_seed2$E/s16 \
  bk12_s0=$M/v7h_gbk12_w2$E/s16 bk12_s1=$M/v7h_gbk12_w2_seed1$E/s16 bk12_s2=$M/v7h_gbk12_w2_seed2$E/s16 \
  composed_s0=$M/v7h_gbk12s10k_w1p5$E/s16 composed_s1=$M/v7h_gbk12s10k_w1p5_seed1$E/s16 composed_s2=$M/v7h_gbk12s10k_w1p5_seed2$E/s16 \
  cfg7=$M/v7h_cfg7$E/s16 cfg10=$M/v7h_cfg10$E/s16 uncond0=$M/v7h_uncond0$E/s16 bk12only=$M/v7h_bk12only$E/s16 \
  bk24_w2=$M/v7h_gbk24_w2$E/s16 bk64_w2=$M/v7h_gbk64_w2$E/s16 autog_w2=$M/v7h_autog10k_w2$E/s16 bk12_w3=$M/v7h_gbk12_w3$E/s16 \
  composed_ddpm50=$M/v7h_stk_w1p5_ddpm50$E/s16 v7lr_cfg4_s1=$M/v7lr_seed1$E/s16 v7lr_bk12_s1=$M/v7lr_gbk12_w2_seed1$E/s16
echo "[$(date +%m%d-%H:%M)] CLIP 12"
clip 12 real=$M/heldout3000_totensor_s12 cfg4_s0=$M/v7h_r12_cfg4$E/s12 cfg4_s1=$M/v7h_r12_cfg4_seed1$E/s12 \
  autog_s0=$M/v7h_r12_autog10k_w1p5$E/s12 autog_s1=$M/v7h_r12_autog10k_w1p5_seed1$E/s12 bk16rev=$M/v7h_r12_bk16_w2$E/s12
echo "[$(date +%m%d-%H:%M)] CLIP 20"
clip 20 real=$M/heldout3000_totensor_s20 cfg4_s0=$M/v7h_r20_cfg4$E/s20 cfg4_s1=$M/v7h_r20_cfg4_seed1$E/s20 \
  bk16_s0=$M/v7h_r20_bk16_w2$E/s20 bk16_s1=$M/v7h_r20_bk16_w2_seed1$E/s20 bk12=$M/v7h_r20_bk12_w2$E/s20 \
  autog_s0=$M/v7h_r20_autog10k_w1p5$E/s20 autog_s1=$M/v7h_r20_autog10k_w1p5_seed1$E/s20 \
  composed_s0=$M/v7h_r20_bk16s10k_w1p5$E/s20 composed_s1=$M/v7h_r20_bk16s10k_w1p5_seed1$E/s20 \
  bk24rev=$M/v7h_r20_bk24_w2$E/s20 bk32rev=$M/v7h_r20_bk32_w2$E/s20
echo "[$(date +%m%d-%H:%M)] CLIP 24"
clip 24 real=$M/heldout3000_totensor_s24 cfg4_s0=$M/v7h_r24_cfg4$E/s24 cfg4_s1=$M/v7h_r24_cfg4_seed1$E/s24 \
  bk16_s0=$M/v7h_r24_bk16_w2$E/s24 bk16_s1=$M/v7h_r24_bk16_w2_seed1$E/s24 bk12=$M/v7h_r24_bk12_w2$E/s24 bk20=$M/v7h_r24_bk20_w2$E/s24 \
  autog_s0=$M/v7h_r24_autog10k_w1p5$E/s24 autog_s1=$M/v7h_r24_autog10k_w1p5_seed1$E/s24 \
  composed_s0=$M/v7h_r24_bk16s10k_w1p5$E/s24 composed_s1=$M/v7h_r24_bk16s10k_w1p5_seed1$E/s24 bk32rev=$M/v7h_r24_bk32_w2$E/s24
echo "[$(date +%m%d-%H:%M)] CLIP 32"
clip 32 real=$M/heldout3000_totensor_s32 cfg4=$M/v7h_r32_cfg4$E/s32 bk24=$M/v7h_r32_bk24_w2$E/s32 bk16=$M/v7h_r32_bk16_w2$E/s32 \
  autog=$M/v7h_r32_autog10k_w1p5$E/s32 composed=$M/v7h_r32_bk24s10k_w1p5$E/s32 bk48rev=$M/v7h_r32_bk48_w2$E/s32
echo "[$(date +%m%d-%H:%M)] q16 @20/24/32"
for t in v7h_r20_cfg4 v7h_r20_bk16_w2 v7h_r20_autog10k_w1p5 v7h_r20_bk16s10k_w1p5; do bash baseline/q16_eval_r.sh ${t}_matched 20 $GPU; done
for t in v7h_r24_cfg4 v7h_r24_bk16_w2 v7h_r24_autog10k_w1p5 v7h_r24_bk16s10k_w1p5; do bash baseline/q16_eval_r.sh ${t}_matched 24 $GPU; done
for t in v7h_r32_cfg4 v7h_r32_bk24_w2 v7h_r32_autog10k_w1p5 v7h_r32_bk24s10k_w1p5; do bash baseline/q16_eval_r.sh ${t}_matched 32 $GPU; done
echo DIAG_CLIPQ16_DONE
