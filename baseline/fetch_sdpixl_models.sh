#!/bin/bash
# Re-fetch every weight SD-piXL loads.  Another user's disk cleanup removed ~/.cache/huggingface in
# mid-September; HF_HUB_OFFLINE=1 then reports the loss as "<repo> does not appear to have a file
# named config.json", which reads like a bad model name rather than a missing cache -- and it reports
# them one at a time, so fetch the whole set rather than chasing them run by run.
# The .bin exclusion keeps the PyTorch duplicates of the safetensors out; taesdxl ships only .bin.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
BIG="stabilityai/stable-diffusion-xl-base-1.0 diffusers/controlnet-canny-sdxl-1.0-mid \
     diffusers/controlnet-depth-sdxl-1.0-mid madebyollin/sdxl-vae-fp16-fix"
SMALL="madebyollin/taesdxl"
for r in $SMALL; do
  for try in 1 2 3; do $HF download "$r" && break; sleep 10; done
done
for r in $BIG; do
  for try in 1 2 3 4 5; do
    echo "[$r try $try]"
    $HF download "$r" --exclude "*.onnx" "*.onnx_data" "*.msgpack" "*.bin" && break
    sleep 10
  done
done
echo SDPIXL_MODELS_DONE
du -sh ~/.cache/huggingface/hub
