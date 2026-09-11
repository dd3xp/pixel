#!/bin/bash
# SDXL generality pilot (node09): does a small `original_size` with the SAME caption work as a guidance reference?
# 1000 COCO-30k val2014 captions (shard 0, rows 0-999), 1024 px, Euler 30 steps, identical per-prompt noise in
# every config. Scored afterwards with FD-DINOv2 (ViT-L/14) against the matching real COCO images + CLIP score.
# Key comparison first: CFG w5 (baseline) / negos (diffusers `negative_original_size` practice) / cfglabel (ours).
# Rules on node09 (user 09-12): only GPUs 4-7 are ours, midi keeps 0-3; wait if another job already holds > 40 GB.
# Usage: bash baseline/run_sdxl_pilot.sh <gpu> <queue: a|b>
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=$1; Q=$2; export CUDA_VISIBLE_DEVICES=$GPU
PROMPTS=data/coco30k/captions_shard0.txt
if [ "$Q" = a ]; then
  CFGS=("cfg5 --mode cfg --w 5" "negos512 --mode negos --w 5 --low 512" "cfg3 --mode cfg --w 3"
        "cfg7 --mode cfg --w 7" "label512_w3 --mode label --w 3 --low 512")
else
  CFGS=("cfglabel512_l1 --mode cfglabel --w 5 --lam 1 --low 512" "cfglabel512_l0p5 --mode cfglabel --w 5 --lam 0.5 --low 512"
        "cfglabel256_l1 --mode cfglabel --w 5 --lam 1 --low 256" "label512_w2 --mode label --w 2 --low 512"
        "negos256 --mode negos --w 5 --low 256")
fi
for c in "${CFGS[@]}"; do
  tag=${c%% *}; args=${c#* }
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -gt 40000 ]; do sleep 60; done
  echo "[$(date +%m%d-%H:%M)] GPU$GPU $tag :: $args"
  $P src/sdxl_gen/sdxl_sizeguide.py --prompts $PROMPTS --out runs_out/sdxl_pilot/$tag --end 1000 --bs 4 $args \
    || { echo "SDXL_PILOT_FAIL $tag"; touch logs/sdxl_pilot_$Q.FAILING; }
done
echo SDXL_PILOT_${Q}_DONE
