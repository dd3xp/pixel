#!/bin/bash
# Oracle diagnostic for probe_mdm: hand the model the real ALPHA channel (the silhouette) and let it decode RGB.
# probe_mdm is a strong inpainter (80% exact tokens at 90% masking) that collapses from an all-masked canvas,
# because the marginal mode is "transparent" and the BLIP captions give no anchor. If FD drops sharply once a
# structural anchor exists, the failure is the anchor, not the discrete representation -- which decides whether
# the family gets closed or re-judged after recaptioning.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
run() { local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  $P src/v6/sample_mdm.py --ckpt workdir/probe_mdm/model_latest.pt --size 16 --out runs_out/${tag}_matched_eval "$@" \
    && $P src/v6/fd_fair.py --size 16 --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16.json \
    && touch runs_out/${tag}_matched_eval/.done
}
run mdm_oracle_alpha       --steps 32 --edit_rounds 2 --context_alpha runs_out/heldout3000_totensor_s16
run mdm_oracle_alpha_greedy --steps 32 --edit_rounds 2 --greedy --context_alpha runs_out/heldout3000_totensor_s16
echo MDM_ORACLE_DONE
