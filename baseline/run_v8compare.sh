#!/bin/bash
# Judge the 09-16 data-policy pilots against their own control.
# Control = workdir/v7r/model_step020000.pt: identical network, steps, captions and sampler; only the
# target distribution differs (v7r trains 48 px art into the 16 px bucket, keeps duplicate copies, and
# never quantises).  Both FD protocols are reported: the default reference (all real sprites, mostly
# downscaled artwork) and --native (only sprites that are natively <= 16 px, i.e. real pixel art).
# Usage: bash baseline/run_v8compare.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ev() {  # <tag> <ckpt> <cfg>
  local tag=$1 ck=$2 w=$3
  if [ ! -f runs_out/${tag}_matched_eval/.done ]; then
    CKPT=$ck EXTRA="--cfg $w --chunk 500" PROMPTS=runs_out/heldout3000_prompts_recap.txt \
      bash baseline/eval_matched_r.sh $tag 16 $GPU || return 1
    touch runs_out/${tag}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16_native.json
}
ev v7r_ctrl20k_cfg1p5 workdir/v7r/model_step020000.pt 1.5
ev v8a_cfg1p5 workdir/v8a/model_latest.pt 1.5
ev v8b_cfg1p5 workdir/v8b/model_latest.pt 1.5
echo V8COMPARE_DONE
