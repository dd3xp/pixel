#!/bin/bash
# Bundle prompt-aligned samples (prompts 0..N-1, seed 0) of the paper's qualitative-figure configs at every resolution
# into /tmp/kw_qual.tgz (run on the server; pulled by paper_assets/make_qual_r.py workflow).  Usage: bash baseline/pull_qual.sh [N]
N=${1:-24}
cd /mnt/data/kw/RoundSquisheen/pixel/pixel/runs_out
L=""
for i in $(seq 0 $((N-1))); do
  s=$(printf "%02d_0.png" $i); r=$(printf "%05d.png" $i)
  for R in 12 16 20 24 32; do L="$L heldout3000_totensor_s$R/$r"; done
  for d in v7h v7h_autog10k_w1p5 v7h_gbk12_w2 v7h_gbk12s10k_w1p5 v7h_bk12only v7h_gbk24_w2; do L="$L ${d}_matched_eval/s16/$s"; done
  for d in v7h_r12_cfg4 v7h_r12_autog10k_w1p5 v7h_r12_bk16_w2; do L="$L ${d}_matched_eval/s12/$s"; done
  for d in v7h_r20_cfg4 v7h_r20_bk16_w2 v7h_r20_autog10k_w1p5 v7h_r20_bk16s10k_w1p5 v7h_r20_bk16only v7h_r20_bk32_w2; do L="$L ${d}_matched_eval/s20/$s"; done
  for d in v7h_r24_cfg4 v7h_r24_bk16_w2 v7h_r24_autog10k_w1p5 v7h_r24_bk16s10k_w1p5 v7h_r24_bk16only v7h_r24_bk32_w2; do L="$L ${d}_matched_eval/s24/$s"; done
  for d in v7h_r32_cfg4 v7h_r32_bk24_w2 v7h_r32_autog10k_w1p5 v7h_r32_bk24s10k_w1p5 v7h_r32_bk48_w2; do L="$L ${d}_matched_eval/s32/$s"; done
done
tar czf /tmp/kw_qual.tgz $L && ls -la /tmp/kw_qual.tgz
