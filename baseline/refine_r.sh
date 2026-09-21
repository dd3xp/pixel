#!/bin/bash
# The post-process at the other two resolutions, so the plug-in claim is not a 16 px result.  Each
# resolution uses the k the sweep found best there (20 px k=8, 24 px k=6), matching the external table.
# Usage: bash baseline/refine_r.sh <gpu>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
GPU=${1:-6}
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROMPTS=runs_out/ext200/v8n/icg2_pal4/prompts.txt
run() {   # <R> <k> <system> <input dir>
  local R=$1 K=$2 SYS=$3 SRC=$4
  for st in 0.4 0.6; do
    local t=refine${R}_${SYS}_s${st/./p}
    local d=runs_out/refine/${t}
    if [ ! -f $d/.done ]; then
      CUDA_VISIBLE_DEVICES=$GPU $P src/v6/sample_e.py --ckpt workdir/v8n/model_latest.pt \
        --buckets 12,16,20,24,32,48,64 --sizes $R --prompts $PROMPTS --n 1 --seed 0 \
        --guide_mode icg --cfg 2 --pal $K --chunk 200 \
        --init_dir $SRC --init_strength $st --out $d && touch $d/.done || continue
    fi
    CUDA_VISIBLE_DEVICES=$GPU $P src/v6/fd_fair.py --size $R --native \
      --gen $d/s$R --out runs_out/fair_fd${R}_native.json
  done
  $P src/v6/refine_fidelity.py --init $SRC --size $R \
    --runs runs_out/refine/refine${R}_${SYS}_s0p4/s$R runs_out/refine/refine${R}_${SYS}_s0p6/s$R
}
run 20 8 gpt runs_out/ext200/q/gpt20_q8/s20
run 24 6 gpt runs_out/ext200/q/gpt24_q6/s24
echo REFINE_R_DONE
