#!/bin/bash
# Score the full-length model on the new corpus the way the paper reports v8n: the main configuration
# (ICG w=2 + 4-colour projection) over three seeds, so the headline number carries its own spread,
# plus k=6 as the balanced setting.  Waits for run_v8full.sh to finish its two CFG evals first.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ ! -f runs_out/v8p_cfg2_matched_eval/.done ]; do sleep 120; done
run() {   # <tag> <extra>
  local t=$1; shift
  if [ ! -f runs_out/${t}_matched_eval/.done ]; then
    CKPT=workdir/v8p/model_latest.pt EXTRA="$* --chunk 500" \
      PROMPTS=runs_out/heldout3000_prompts_recap.txt bash baseline/eval_matched_r.sh $t 16 $GPU \
      && touch runs_out/${t}_matched_eval/.done
  fi
  CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
    --gen runs_out/${t}_matched_eval/s16 --out runs_out/fair_fd16_native.json
}
run v8p_icg2_pal4        --guide_mode icg --cfg 2 --pal 4
run v8p_icg2_pal4_seed1  --guide_mode icg --cfg 2 --pal 4 --seed 1
run v8p_icg2_pal4_seed2  --guide_mode icg --cfg 2 --pal 4 --seed 2
run v8p_icg2_pal6        --guide_mode icg --cfg 2 --pal 6
run v8p_icg2             --guide_mode icg --cfg 2
echo AFTER_V8P_DONE
