#!/bin/bash
# Mechanism check for cross-resolution autoguidance: sample the *pure* weak references at 16 px (w=0 -> x follows
# e_u alone) and compare their simplicity statistics (stats_simplicity.py) and FD with real natives / block-averaged
# reals.  bk12only = bucket:12 belief, bk24only / bk64only = higher-bucket beliefs, snap10k = early-snapshot model,
# uncond0 = text-unconditional.  Runs after dres20.  Usage: bash baseline/diag_mech.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
until grep -q DIAG_RES20_DONE logs/diag_res20.log 2>/dev/null; do sleep 120; done
CK=workdir/v7h/model_latest.pt
WK=workdir/v7h/model_step010000.pt
run() {  # <tag> <extra sample_e args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  CKPT=$CK EXTRA="$*" bash baseline/eval_matched_r.sh $tag 16 $GPU && touch runs_out/${tag}_matched_eval/.done
}
run v7h_bk12only --cfg 0 --guide_mode bucket:12
run v7h_bk24only --cfg 0 --guide_mode bucket:24
run v7h_snap10konly --cfg 0 --guide_ckpt $WK
run v7h_uncond0 --cfg 0
run v7h_bk64only --cfg 0 --guide_mode bucket:64
echo "[$(date +%m%d-%H:%M)] fd_decomp + stats"
$P src/v6/fd_decomp.py --size 16 --gen $(ls -d runs_out/v7h_*only_matched_eval/s16) runs_out/v7h_uncond0_matched_eval/s16
CUDA_VISIBLE_DEVICES= $P src/v6/stats_simplicity.py --side 16 --real_bucket 12 16 24 --block real16:2 --dirs \
  real16=runs_out/ref_totensor_s16 v7h_cfg4=runs_out/v7h_matched_eval/s16 \
  bk12only=runs_out/v7h_bk12only_matched_eval/s16 bk24only=runs_out/v7h_bk24only_matched_eval/s16 \
  bk64only=runs_out/v7h_bk64only_matched_eval/s16 snap10konly=runs_out/v7h_snap10konly_matched_eval/s16 \
  uncond0=runs_out/v7h_uncond0_matched_eval/s16 autog10k_w1p5=runs_out/v7h_autog10k_w1p5_matched_eval/s16 \
  bk12_w2=runs_out/v7h_gbk12_w2_matched_eval/s16 stack_w1p5=runs_out/v7h_gbk12s10k_w1p5_matched_eval/s16
echo DIAG_MECH_DONE
