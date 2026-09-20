#!/bin/bash
# Fetch the 2D art half of the OpenGameArt CC0 bundle (nyuuzyou/OpenGameArt-CC0, ~23 GB of the 122 GB
# repository; the 3D/music/texture/sound halves are irrelevant here).  This is the one OGA licence
# bundle we never processed: the 09-16 rebuild used CC-BY-4.0, CC-BY-SA-3.0 and OGA-BY-3.0.  CC0 is
# also the cleanest licence for a released corpus, like the Kenney packs.
# Usage: bash baseline/fetch_oga_cc0.sh
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
$HF download nyuuzyou/OpenGameArt-CC0 --repo-type dataset \
  --include "2D_Art*.zip" "2D_Art.jsonl.zst" "ATTRIBUTION.md" "archive_index.csv" "README.md" \
  --local-dir data/OpenGameArt-CC0 --max-workers 4
echo OGA_CC0_FETCH_DONE
du -sh data/OpenGameArt-CC0
