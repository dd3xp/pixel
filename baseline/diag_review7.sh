#!/bin/bash
# Fill the last numeric [TBD]s of draft_full.md (no new sampling): (a) q16 of the snaplo composed row (seed 0); (b) fd_decomp
# P/R/D/C of the composed row seed 1 (Table A6); (c) simplicity statistics (colours / flat ratio / TV) of the best-CFG samples
# so Sec. 5 can compare the CFG optimum, not only w=4, against the guided rows.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review7 3 bash baseline/diag_review7.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
M=runs_out; E=_matched_eval
[ -f $M/v7h_stk_w1p5_snaplo_matched_q16/.done ] || { bash baseline/q16_eval.sh v7h_stk_w1p5_snaplo_matched $GPU && touch $M/v7h_stk_w1p5_snaplo_matched_q16/.done; }
$P src/v6/fd_decomp.py --size 16 --gen $M/v7h_gbk12s10k_w1p5_seed1$E/s16 $M/v7h_gbk12s10k_w1p5_seed2$E/s16 $M/v7h_cfg1p5$E/s16 $M/v7h_cfg1p5_seed1$E/s16 $M/v7h_cfg1p5_seed2$E/s16
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 16 --real_bucket 16 --dirs \
  real16=$M/ref_totensor_s16 v7h_cfg4=$M/v7h$E/s16 v7h_cfg1p5=$M/v7h_cfg1p5$E/s16 v7h_cfg2=$M/v7h_cfg2$E/s16 v7h_cfg3=$M/v7h_cfg3$E/s16 \
  v7h_cfg1=$M/v7h_cfg1$E/s16 pagmid_w2=$M/v7h_pagmid_w2$E/s16 bk12_w2=$M/v7h_gbk12_w2$E/s16 stack_w1p5=$M/v7h_gbk12s10k_w1p5$E/s16 bk12only=$M/v7h_bk12only$E/s16
echo DIAG_REVIEW7_DONE
