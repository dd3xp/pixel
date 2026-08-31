#!/bin/bash
# Name the 8 already-successful runs the same way the worker names new ones,
# so the queue's "already done" check sees them.  Mapping is by launch order
# recorded in logs/base10k_{a,b}.log.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/sdpixl_db32_10k || exit 1
R=/mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/results
L=/mnt/data/kw/RoundSquisheen/pixel/pixel/baseline/locks
mkdir -p "$R" "$L"
map() { [ -f "$1/final_argmax.png" ] && cp "$1/final_argmax.png" "$R/10k_s$2_p$3.png" && mkdir -p "$L/$2_$3" && echo "  s$2 p$3 <- $1"; }
map 2026-08-27-10-27-46-sdxl-im12x12-dawnbringer32 12 1
map 2026-08-27-13-55-02-sdxl-im12x12-dawnbringer32 12 2
map 2026-08-27-17-21-45-sdxl-im12x12-dawnbringer32 12 3
map 2026-08-27-20-51-31-sdxl-im12x12-dawnbringer32 12 4
map 2026-08-28-00-21-12-sdxl-im12x12-dawnbringer32 12 5
map 2026-08-27-10-27-46-sdxl-im20x20-dawnbringer32 20 1
map 2026-08-27-15-18-40-sdxl-im20x20-dawnbringer32 20 2
map 2026-08-27-21-15-57-sdxl-im20x20-dawnbringer32 20 3
echo "backfilled: $(ls $R | wc -l)"
