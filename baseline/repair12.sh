#!/bin/bash
# rerun the three 12px prompts whose outputs were lost to the full disk / HF timeouts
cd /mnt/data/kw/RoundSquisheen/pixel/SD-piXL
source /mnt/data/kw/anaconda3/etc/profile.d/conda.sh && conda activate SD-piXL
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=60
for p in 'a pixel art gold ingot' 'a pixel art iron ingot' 'a pixel art blue diamond gem'; do
  echo "[$(date +%m%d-%H:%M)] repair12: $p"
  python main.py -c /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/sdpixl_db32_10k.yaml --size=12,12 -pt "$p" 2>&1 | grep -E 'Error|Traceback' | tail -2
done
echo REPAIR12_DONE
