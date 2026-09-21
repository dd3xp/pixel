"""How native is each corpus's 16 px bucket?

v8p (the new corpus) beats v8n at 20k steps and loses to it at 80k on the reported configuration.  The
suspicion is dilution: the CC-BY-3.0 bundle is large but only 903 of its 13,482 sprites are natively
16 px or smaller, so most of what it adds to the 16 px bucket is larger art downscaled into it -- the
exact defect this paper is about.  This measures that directly from the manifests.
"""
import csv
from collections import Counter
from pathlib import Path

MAX_DOWN = 1.5
BUCKETS = [12, 16, 20, 24, 32, 48, 64]

SETS = {
    "oga3 (v8n new sprites)": ("data/oga3_manifest.csv", "data/oga3_keep.txt"),
    "oga5 (CC-BY-3.0)":       ("data/oga5_clean_manifest.csv", "data/oga5_keep.txt"),
}


def rows(man, keep):
    ks = None
    if keep and Path(keep).exists():
        ks = {l.strip() for l in open(keep, encoding="utf-8") if l.strip()}
    out = []
    for r in csv.DictReader(open(man, encoding="utf-8")):
        if ks is None or r["path"] in ks:
            out.append(int(r["side"]))
    return out


for name, (man, keep) in SETS.items():
    if not Path(man).exists():
        print(f"{name}: missing {man}")
        continue
    sides = rows(man, keep)
    feeds16 = [s for s in sides if s <= 16 * MAX_DOWN]
    native16 = [s for s in feeds16 if s <= 16]
    print(f"{name}: {len(sides)} sprites")
    print(f"  feed the 16 px bucket (side <= 24): {len(feeds16)}")
    print(f"  of those natively <= 16 px:         {len(native16)} "
          f"({100 * len(native16) / max(1, len(feeds16)):.1f}%)")
    print(f"  size histogram of the feeders: {sorted(Counter(feeds16).items())[:14]}")
