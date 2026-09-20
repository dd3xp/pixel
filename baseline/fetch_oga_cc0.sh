#!/bin/bash
# Fetch the 2D art half of the OpenGameArt CC0 bundle (nyuuzyou/OpenGameArt-CC0, ~23 GB of the 122 GB
# repository; the 3D/music/texture/sound halves are irrelevant here).  This is the one OGA licence
# bundle we never processed: the 09-16 rebuild used CC-BY-4.0, CC-BY-SA-3.0 and OGA-BY-3.0.  CC0 is
# also the cleanest licence for a released corpus, like the Kenney packs.
# The mirror drops connections part-way through the larger zips, so retry until every part is on disk
# (hf resumes from the .incomplete file) rather than trusting one pass.
# Usage: bash baseline/fetch_oga_cc0.sh
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
missing() { local m=0; for i in $(seq -w 0 28); do [ -f data/OpenGameArt-CC0/2D_Art_$i.zip ] || m=$((m+1)); done; echo $m; }
for try in $(seq 1 30); do
  n=$(missing); echo "[try $try] missing $n/29 zips"
  [ "$n" = 0 ] && break
  $HF download nyuuzyou/OpenGameArt-CC0 --repo-type dataset \
    --include "2D_Art*.zip" "2D_Art.jsonl.zst" "ATTRIBUTION.md" "archive_index.csv" "README.md" \
    --local-dir data/OpenGameArt-CC0 --max-workers 2 || true
  sleep 10
done
echo "OGA_CC0_FETCH_FINAL missing=$(missing)"
du -sh data/OpenGameArt-CC0
