#!/bin/bash
# Same fetch, but with a download timeout: the mirror hung at 9/10 files on the canny controlnet and
# a hung connection never returns, so the retry loop around it never fired.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 HF_HUB_DOWNLOAD_TIMEOUT=30
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
for r in madebyollin/taesdxl madebyollin/sdxl-vae-fp16-fix \
         diffusers/controlnet-canny-sdxl-1.0-mid diffusers/controlnet-depth-sdxl-1.0-mid \
         stabilityai/stable-diffusion-xl-base-1.0; do
  for try in 1 2 3 4 5 6; do
    echo "[$r try $try]"
    timeout 1800 $HF download "$r" --exclude "*.onnx" "*.onnx_data" "*.msgpack" && break
    sleep 5
  done
done
echo SDPIXL_MODELS_READY
