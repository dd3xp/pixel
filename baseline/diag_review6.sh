#!/bin/bash
# The alignment-fidelity frontier was only established at 16 px.  diag_review5 showed the best-CFG rows at 20/24/32 px still
# have +0.55..0.65 CLIP over the composed row, so (a) score the lower-w CFG rows that already exist at 20/24/32 px (CFG w=1.5;
# defines the CFG curve's low-alignment end there) and (b) run the alignment-recovering variant composed∘bucketu:<R'> w=1.5 at
# 20/24/32 px (bucketu = wrong bucket AND unconditional text; 2 NFE, same cost) + its CLIP, to test whether the 16 px
# "dominates the best-CFG point in both coordinates" result transfers.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review6 3 bash baseline/diag_review6.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt; WK=workdir/v7h/model_step010000.pt
runR() {  # <tag> <R> <extra sample_e args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
runR v7h_r20_bku16s10k_w1p5 20 --cfg 1.5 --guide_mode bucketu:16 --guide_ckpt $WK
runR v7h_r24_bku16s10k_w1p5 24 --cfg 1.5 --guide_mode bucketu:16 --guide_ckpt $WK
runR v7h_r32_bku24s10k_w1p5 32 --cfg 1.5 --guide_mode bucketu:24 --guide_ckpt $WK
runR v7h_r12_bku16s10k_w1p5 12 --cfg 1.5 --guide_mode bucketu:16 --guide_ckpt $WK   # 12 px has no lower bucket: is the text-drop alone useful?
M=runs_out; E=_matched_eval
$P src/v6/clip_score.py --size 20 --out runs_out/clip_scores.json --dirs r20_cfg1p5=$M/v7h_r20_cfg1p5$E/s20 r20_cfg2=$M/v7h_r20_cfg2$E/s20 r20_bku16s10k=$M/v7h_r20_bku16s10k_w1p5$E/s20
$P src/v6/clip_score.py --size 24 --out runs_out/clip_scores.json --dirs r24_cfg1p5=$M/v7h_r24_cfg1p5$E/s24 r24_cfg3=$M/v7h_r24_cfg3$E/s24 r24_bku16s10k=$M/v7h_r24_bku16s10k_w1p5$E/s24
$P src/v6/clip_score.py --size 32 --out runs_out/clip_scores.json --dirs r32_cfg1p5=$M/v7h_r32_cfg1p5$E/s32 r32_cfg3=$M/v7h_r32_cfg3$E/s32 r32_bku24s10k=$M/v7h_r32_bku24s10k_w1p5$E/s32
$P src/v6/clip_score.py --size 12 --out runs_out/clip_scores.json --dirs r12_cfg1p5=$M/v7h_r12_cfg1p5$E/s12 r12_cfg3=$M/v7h_r12_cfg3$E/s12 r12_bku16s10k=$M/v7h_r12_bku16s10k_w1p5$E/s12
$P src/v6/fid_kid_fair.py --size 20 --out runs_out/incep_scores.json --gen $M/v7h_r20_bku16s10k_w1p5$E/s20
$P src/v6/fid_kid_fair.py --size 24 --out runs_out/incep_scores.json --gen $M/v7h_r24_bku16s10k_w1p5$E/s24
$P src/v6/fid_kid_fair.py --size 32 --out runs_out/incep_scores.json --gen $M/v7h_r32_bku24s10k_w1p5$E/s32
echo DIAG_REVIEW6_DONE
