#!/bin/bash
# The remaining OpenGameArt licence bundles with a licence we are willing to release under:
# CC-BY-3.0 (4.0 GB of 2D art) and CC-BY-SA-4.0 (0.7 GB).  GPL-2.0/3.0 and Mixed-Licenses are skipped
# on purpose -- copyleft code licences on art and unresolved per-asset terms are not worth the trouble
# for a corpus we intend to publish.  Retries until every part is on disk, like fetch_oga_cc0.sh.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export HF_ENDPOINT=https://hf-mirror.com PYTHONNOUSERSITE=1
HF=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/hf
get() {  # <repo suffix> <local dir> <expected 2D zips>
  local repo=$1 dir=$2 want=$3
  for try in $(seq 1 30); do
    n=$(ls data/$dir/2D_Art_*.zip 2>/dev/null | wc -l)
    echo "[$dir try $try] have $n/$want zips"
    [ "$n" -ge "$want" ] && return 0
    $HF download nyuuzyou/$repo --repo-type dataset \
      --include "2D_Art*.zip" "2D_Art.jsonl.zst" "ATTRIBUTION.md" "archive_index.csv" \
      --local-dir data/$dir --max-workers 2 || true
    sleep 10
  done
  return 1
}
get OpenGameArt-CC-BY-3.0    OpenGameArt-CC-BY-3.0    10
get OpenGameArt-CC-BY-SA-4.0 OpenGameArt-CC-BY-SA-4.0 1
echo "OGA_REST_FETCH_FINAL ccby3=$(ls data/OpenGameArt-CC-BY-3.0/2D_Art_*.zip 2>/dev/null | wc -l) ccbysa4=$(ls data/OpenGameArt-CC-BY-SA-4.0/2D_Art_*.zip 2>/dev/null | wc -l)"
du -sh data/OpenGameArt-CC-BY-3.0 data/OpenGameArt-CC-BY-SA-4.0
