#!/bin/bash
# After captioning: classify the new CC-BY-3.0 captions by content, drop glyph/junk and unusable
# captions, and write the training CSV.  Same steps the Kenney corpus went through on 09-20.
cd /mnt/data/kw/RoundSquisheen/pixel/pixel
export PYTHONNOUSERSITE=1 PIXEL_API_BASE=http://113.45.39.247:3001/v1
P=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
while tmux has-session -t v10cap 2>/dev/null; do sleep 60; done
echo "captions: $(wc -l < runs/oga5_captions.jsonl)"
$P - <<'PY'
import csv, json
rows = [json.loads(l) for l in open("runs/oga5_captions.jsonl", encoding="utf-8")]
with open("data/oga5_captions_recap.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "text"])
    for r in rows:
        w.writerow([r["path"], r["text"]])
print("csv rows", len(rows))
PY
$P baseline/classify_captions.py --csv data/oga5_captions_recap.csv --out runs/oga5_capcls.jsonl
echo OGA5_PIPELINE_DONE
