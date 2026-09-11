#!/bin/bash
# EVAL for probe_mdm (A1): confidence-ordered decoding, then the standard matched FD.
# Decision (arch_scout): bare FD <=12 -> mechanism real, add cross-resolution guidance and chase 7.53;
#                        12-20 -> sweep decoding steps / temperature; >=20 -> close the discrete family.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${1:-2}; export CUDA_VISIBLE_DEVICES=$GPU
CK=workdir/probe_mdm/model_latest.pt
run() {  # <tag> <extra sample_mdm args>
  local tag=$1; shift
  [ -f runs_out/${tag}_matched_eval/.done ] && return
  echo "[$(date +%m%d-%H:%M)] RUN $tag :: $*"
  $P src/v6/sample_mdm.py --ckpt $CK --size 16 --out runs_out/${tag}_matched_eval "$@" \
    && $P src/v6/fd_fair.py --size 16 --gen runs_out/${tag}_matched_eval/s16 --out runs_out/fair_fd16.json \
    && touch runs_out/${tag}_matched_eval/.done
}
run mdm_s32_e2        --steps 32 --edit_rounds 2
run mdm_s64_e2        --steps 64 --edit_rounds 2
run mdm_s32_e0        --steps 32 --edit_rounds 0
run mdm_s32_e2_t0p8   --steps 32 --edit_rounds 2 --temp 0.8
echo EVAL_MDM_DONE
