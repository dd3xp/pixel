#!/bin/bash
# The composed-bucketu rows became headline numbers (32 px 65.27 now beats plain composed 69.27) but were run at seed 0 only.
# Run seeds 1 and 2 at 20 / 24 / 32 px so every headline row in Tables 3-4 is a 3-seed mean, like the CFG and composed rows.
# Launch: NEED_MB=8000 setsid nohup bash supervise.sh diag_review8 3 bash baseline/diag_review8.sh 3 >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
GPU=${1:-3}; export CUDA_VISIBLE_DEVICES=$GPU
H=workdir/v7h/model_latest.pt; WK=workdir/v7h/model_step010000.pt
runR() {  # <tag> <R> <extra sample_e args>
  local tag=$1 R=$2; shift 2
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag @$R :: $*"
  CKPT=$H EXTRA="$*" bash baseline/eval_matched_r.sh $tag $R $GPU && touch runs_out/${tag}_matched_eval/.done
}
for s in 1 2; do
  runR v7h_r20_bku16s10k_w1p5_seed$s 20 --cfg 1.5 --guide_mode bucketu:16 --guide_ckpt $WK --seed $s
  runR v7h_r24_bku16s10k_w1p5_seed$s 24 --cfg 1.5 --guide_mode bucketu:16 --guide_ckpt $WK --seed $s
  runR v7h_r32_bku24s10k_w1p5_seed$s 32 --cfg 1.5 --guide_mode bucketu:24 --guide_ckpt $WK --seed $s
done
echo DIAG_REVIEW8_DONE
