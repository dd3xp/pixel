#!/bin/bash
# External baseline SDXL + Pixel Art XL LoRA on the 200 recaptioned prompts: generate 1024 px -> api_to_sprites
# (cutout + to_tensor) at 16/20/24 -> fd_fair with the same reference as runs_out/ext200 (n=200 protocol).
# PixelOE variant is made locally afterwards (pixeloe is installed on the local machine).
# Launch: NEED_MB=16000 setsid nohup bash supervise.sh ext_lora 7 bash baseline/run_ext_lora.sh >/dev/null 2>&1 &
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${CUDA_VISIBLE_DEVICES:-7}
E=runs_out/ext200/runs/ext_lora
$P baseline/sdxl_lora_gen.py --out $E/big --n 200 || { echo EXT_LORA_GEN_FAIL; exit 1; }
for R in 16 20 24; do
  [ "$(ls $E/s$R 2>/dev/null | wc -l)" -ge 200 ] || $P baseline/api_to_sprites.py --big $E/big --out $E --size $R
  $P src/v6/fd_fair.py --size $R --gen $E/s$R --out runs_out/ext200/fd_ext200_s$R.json
done
echo EXT_LORA_DONE
