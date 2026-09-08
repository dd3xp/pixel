#!/bin/bash
# Reviewer-pass (paper_assets/review_pass1.md) cheap experiments, v7h @16 seed 0, matched FD-DINOv2 protocol:
#  R1  CFG swept DOWNWARD (w=1,1.5,2,3) + CLIP  -> matched-alignment / Pareto framing of the headline (bk12 pays -0.5 CLIP)
#  R2  interventional reference: own prediction with x0-contrast shrunk by f (shrink:f, 1 NFE) -> is "low-contrast ref" alone enough?
#  R3  PAG baseline (identity self-attention in mid / mid+d1 layers), alone and with the additive CFG term
#  R4  50-step bare / autog / composed (sampler-step sensitivity)
# Two halves so GPU2 and GPU3 can share the work: PART=A (R1+R2) / PART=B (R3+R4); whichever finishes last does the scoring.
# Launch: PART=A NEED_MB=8000 setsid nohup bash supervise.sh diag_review1a 2 bash baseline/diag_review1.sh 2 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt;  HW=workdir/v7h/model_step010000.pt
PART=${PART:-A}
run16() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @16 :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag 16 $GPU && touch runs_out/${tag}_matched_eval/.done
}
if [ "$PART" = A ]; then
# R1 CFG downward
for w in 1 1p5 2 3; do run16 v7h_cfg${w} --cfg ${w/p/.}; done
# R2 interventional shrink reference (w=2 like bk12; f sweep)
for f in 0p5 0p7 0p85; do run16 v7h_shrink${f}_w2 --cfg 2 --guide_mode shrink:${f/p/.}; done
run16 v7h_shrink0p7_w3 --cfg 3 --guide_mode shrink:0.7
fi
if [ "$PART" = B ]; then
# R3 PAG
for w in 1p5 2 3; do run16 v7h_pagmid_w${w} --cfg ${w/p/.} --guide_mode pag:mid; done
run16 v7h_pagmidd1_w2   --cfg 2 --guide_mode pag:mid,d1
run16 v7h_pagmid_w2_ct1p5 --cfg 2 --guide_mode pag:mid --cfg_text 1.5
# R4 50 steps
run16 v7h_st50_cfg4     --cfg 4   --steps 50
run16 v7h_st50_autog10k --cfg 1.5 --steps 50 --guide_ckpt $HW
run16 v7h_st50_composed --cfg 1.5 --steps 50 --guide_mode bucket:12 --guide_ckpt $HW
fi
ALL="cfg1 cfg1p5 cfg2 cfg3 shrink0p5_w2 shrink0p7_w2 shrink0p85_w2 shrink0p7_w3 pagmid_w1p5 pagmid_w2 pagmid_w3 pagmidd1_w2 pagmid_w2_ct1p5 st50_cfg4 st50_autog10k st50_composed"
for t in $ALL; do [ -f runs_out/v7h_${t}_matched_eval/.done ] || { echo "PART $PART done; $t missing -> other half scores"; echo DIAG_REVIEW1_${PART}_DONE; exit 0; }; done
echo DIAG_REVIEW1_SAMPLED
# CLIP score + R@1 on the new rows (same script as diag_align), then FD decomposition
D=""; for d in $(ls -d runs_out/v7h_cfg*_matched_eval/s16 runs_out/v7h_shrink*_matched_eval/s16 runs_out/v7h_pag*_matched_eval/s16); do
  n=${d#runs_out/v7h_}; D="$D ${n%_matched_eval/s16}=$d"; done
$P src/v6/clip_score.py --size 16 --out runs_out/clip_scores.json --dirs $D
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_cfg*_matched_eval/s16 runs_out/v7h_shrink*_matched_eval/s16 runs_out/v7h_pag*_matched_eval/s16)
echo DIAG_REVIEW1_DONE
