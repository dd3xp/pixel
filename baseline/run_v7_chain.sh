#!/bin/bash
# V7 3-day pipeline on GPU1: train -> MC sample -> FID(12/16/20/24) -> showcase
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_ENDPOINT=https://hf-mirror.com PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=12,16,20,24,32,48,64
echo "[$(date +%m%d-%H:%M)] STAGE train"
$P src/v6/train_v7.py --init_v6 workdir/v6e10_ema/model_latest.pt --steps 60000 --out workdir/v7_lowres || { echo V7_TRAIN_FAIL; exit 1; }
echo "[$(date +%m%d-%H:%M)] STAGE mc"
$P src/v6/sample_e.py --ckpt workdir/v7_lowres/model_latest.pt --buckets $B --prompts prompts/mc_items.txt --sizes 12 16 20 24 --n 4 --out workdir/mc_test_v7
echo "[$(date +%m%d-%H:%M)] STAGE fid"
ls /tmp/inception-2015-12-05.pt >/dev/null 2>&1 || cp data/inception-2015-12-05.pt /tmp/
for S in 12 16 20 24; do $P src/v6/eval_fid.py --ckpt workdir/v7_lowres/model_latest.pt --buckets $B --size $S --n 1000; done
echo "[$(date +%m%d-%H:%M)] STAGE showcase"
$P src/v6/sample_e.py --ckpt workdir/v7_lowres/model_latest.pt --buckets $B --prompts prompts/mc_items.txt --sizes 12 16 20 24 --n 16 --out workdir/showcase_v7
echo V7_CHAIN_DONE
