"""Turn the SDXL pseudo-labels into a training source for v11.

The coverage measurement showed the "draw anything" gap is object VOCABULARY,
not rendering: every output is competent pixel art, it is just often the wrong
object, and the categories that fail (modern electronics, precision tools,
instruments, sportswear) are exactly the ones a game-sprite corpus lacks.  SDXL
has that vocabulary, and the downscale baseline already turns an arbitrary
prompt into a passable 12/16/20/24 sprite -- so it can supply pseudo-labels for
words our corpus never taught.

Only the `mean_raw` variant is used: the sweep over downscale variants found
premultiplied block-mean strongest, and DawnBringer32 quantisation clearly hurt.

train_v7 assigns a bucket from each sprite's pixel size, so writing the four
sizes as separate files puts them in the 12/16/20/24 buckets automatically.
Captions match our own convention ("a pixel art <object>") so the text encoder
sees the same phrasing at train and test time.
"""
import csv
import shutil
from pathlib import Path

import sys

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
# args: [src_run_dir] [vocab_txt] [dst_dir]  (defaults reproduce round 1)
SRC = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs_out/distill_v1")
VOCAB = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "prompts/vocab_distill.txt")
DST = ROOT / (sys.argv[3] if len(sys.argv) > 3 else "data/pseudo")
SIZES = [12, 16, 20, 24]
VARIANT = "mean_raw"

objects = [l.strip() for l in open(VOCAB, encoding="utf-8")
           if l.strip() and not l.startswith("#")]

DST.mkdir(parents=True, exist_ok=True)
rows, missing = [], 0
for i, obj in enumerate(objects):
    slug = obj.replace(" ", "_")
    for n in SIZES:
        src = SRC / f"s{n}" / VARIANT / f"{i:02d}.png"
        if not src.exists():
            missing += 1
            continue
        name = f"{slug}_s{n}.png"
        shutil.copyfile(src, DST / name)
        rows.append((name, f"a pixel art {obj}"))

with open(DST.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "text"])
    w.writerows(rows)

print(f"{len(rows)} pseudo sprites from {len(objects)} objects -> {DST}/ "
      f"({missing} missing)")
print("use: --extra data/pseudo,data/pseudo.csv,<repeat>")
