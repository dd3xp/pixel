#!/bin/bash
# Apply the decoding rule to someone else's sprites.  Quantising a finished external sample helps
# (gpt-image-2: 284 -> 134 native FD) but cannot repair the seams quantisation introduces, which is
# exactly the gap the in-sampler projection closes for our own model.  Here the external sprite is
# re-noised to a fraction of the schedule and denoised with our model under the reported configuration,
# so the projection runs inside a loop that starts from their content.
# Usage: bash baseline/refine_ext.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPTS=runs_out/ext200/v8n/icg2_pal4/prompts.txt
REF=runs_out/ref_native16            # written by fd_ref-style runs; falls back to --native below
for sys in gpt flux2 lora; do
  for st in 0.4 0.6; do
    t=refine_${sys}_s${st/./p}
    d=runs_out/refine/${t}
    if [ ! -f $d/.done ]; then
      CUDA_VISIBLE_DEVICES=$GPU $P src/v6/sample_e.py --ckpt workdir/v8n/model_latest.pt \
        --buckets 12,16,20,24,32,48,64 --sizes 16 --prompts $PROMPTS --n 1 --seed 0 \
        --guide_mode icg --cfg 2 --pal 4 --chunk 200 \
        --init_dir runs_out/ext200/q/${sys}_q4/s16 --init_strength $st \
        --out $d && touch $d/.done || continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size 16 --native \
      --gen $d/s16 --out runs_out/fair_fd16_native.json
  done
done
echo REFINE_EXT_DONE
