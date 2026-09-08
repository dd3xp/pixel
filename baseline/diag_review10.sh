#!/bin/bash
# Two repairs the read-through demands (paper_assets/read_through.md C1 / C3):
#  C1 regime match: every weak-reference belief is sampled unguided, but the "strong model" row it is compared against is
#     CFG w=4.  The matched comparator is the conditional prediction under the CORRECT label, i.e. CFG w=1 (sample_e:
#     e = e_u + w(e_c - e_u) is exactly e_c at w=1).  At 16 px that is already measured (TV 26.2); run it at 20 and 24 px,
#     which also extends the CFG curve downward there, and take simplicity statistics at all three resolutions.
#  C3 gain match: shrink:f with weight w applies the fixed per-step gain f + w(1-f) to the strong x0 contrast.  The earlier
#     sweep used gains of 1.15-1.50, so its divergence may be a step-gain artefact rather than evidence that the label
#     reference's correction is spatially structured.  Sweep the same f at weights giving gains ~1.05-1.15 instead.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review10 3 bash baseline/diag_review10.sh 3 >/dev/null 2>&1 &
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
# --- C1: the regime-matched strong belief (CFG w=1 = pure conditional) at 20 and 24 px
runR v7h_r20_cfg1 20 --cfg 1
runR v7h_r24_cfg1 24 --cfg 1
# --- C3: gain-matched uniform contrast shrink at 16 px.  gain = f + w(1-f):
runR v7h_shrink0p5_w1p1   16 --cfg 1.1  --guide_mode shrink:0.5    # gain 1.05
runR v7h_shrink0p7_w1p15  16 --cfg 1.15 --guide_mode shrink:0.7    # gain 1.045
runR v7h_shrink0p85_w1p33 16 --cfg 1.33 --guide_mode shrink:0.85   # gain 1.05
runR v7h_shrink0p5_w1p3   16 --cfg 1.3  --guide_mode shrink:0.5    # gain 1.15
runR v7h_shrink0p85_w2p5  16 --cfg 2.5  --guide_mode shrink:0.85   # gain 1.225, brackets the earlier w=2 point
M=runs_out; E=_matched_eval
echo "[$(date +%m%d-%H:%M)] simplicity statistics of the regime-matched strong beliefs and the gain-matched shrink rows"
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 20 --dirs real20=$M/ref_totensor_s20 \
  r20_cfg1=$M/v7h_r20_cfg1$E/s20 r20_cfg2p5=$M/v7h_r20_cfg2p5$E/s20 r20_cfg4=$M/v7h_r20_cfg4$E/s20 \
  r20_bk16=$M/v7h_r20_bk16_w2$E/s20 r20_composed=$M/v7h_r20_bk16s10k_w1p5$E/s20
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 24 --dirs real24=$M/ref_totensor_s24 \
  r24_cfg1=$M/v7h_r24_cfg1$E/s24 r24_cfg2=$M/v7h_r24_cfg2$E/s24 r24_cfg4=$M/v7h_r24_cfg4$E/s24 \
  r24_bk16=$M/v7h_r24_bk16_w2$E/s24 r24_composed=$M/v7h_r24_bk16s10k_w1p5$E/s24
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 16 --dirs real16=$M/ref_totensor_s16 \
  shrink0p5_w1p1=$M/v7h_shrink0p5_w1p1$E/s16 shrink0p7_w1p15=$M/v7h_shrink0p7_w1p15$E/s16 \
  shrink0p85_w1p33=$M/v7h_shrink0p85_w1p33$E/s16 shrink0p5_w1p3=$M/v7h_shrink0p5_w1p3$E/s16 \
  shrink0p85_w2p5=$M/v7h_shrink0p85_w2p5$E/s16
echo DIAG_REVIEW10_DONE
