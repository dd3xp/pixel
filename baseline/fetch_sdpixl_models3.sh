#!/bin/bash
# The last two weight sets SD-piXL loads: the depth estimator behind its depth ControlNet, and BLIP-2,
# which it uses to caption the input image.  Fetched together rather than one failed run at a time.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1 HF_HUB_DOWNLOAD_TIMEOUT=30
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
for r in Intel/dpt-hybrid-midas Salesforce/blip2-opt-2.7b; do
  for try in 1 2 3 4 5 6; do
    echo "[$r try $try]"
    timeout 2400 $HF download "$r" --exclude "*.onnx" "*.onnx_data" "*.msgpack" "*.h5" && break
    sleep 5
  done
done
echo SDPIXL_MODELS_READY2
du -sh ~/.cache/huggingface/hub
