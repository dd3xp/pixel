"""The 15.5% leakage figure, recomputed against the canonical hold-out list.

audit_leakage.py rebuilds the split itself and gets 14.6%, which is close enough to look right and not
close enough to be a check.  runs_out/holdout_exclude.txt is the list the training runs actually
exclude, so this uses it: of the 3,000 evaluation sprites, how many have a pixel-identical twin among
the sprites training was allowed to see?
"""
import glob
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def pixhash(p):
    a = np.array(Image.open(p).convert("RGBA"))
    return hashlib.md5(a.tobytes()).hexdigest() + f"_{a.shape[0]}x{a.shape[1]}"


excl = {l.strip().replace("\\", "/") for l in open("runs_out/holdout_exclude.txt", encoding="utf-8") if l.strip()}
allc = sorted({p.replace("\\", "/") for p in
               glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")
               + glob.glob("data/extra_all/**/*.png", recursive=True) + glob.glob("data/extra_all/*.png")})
print(f"corpus {len(allc)}, exclude list {len(excl)}")

# the evaluation sprites are the ones the prompt file was built from: the first 3,000 of the exclude list
held = [p for p in allc if p in excl][:3000]
train = [p for p in allc if p not in excl]
print(f"held {len(held)}, train {len(train)}")

th = Counter()
for p in train:
    try:
        th[pixhash(p)] += 1
    except Exception:
        pass
twin = 0
for p in held:
    try:
        if th.get(pixhash(p), 0):
            twin += 1
    except Exception:
        pass
print(f"held-out sprites with an exact twin in training: {twin}/{len(held)} = {100 * twin / len(held):.1f}%  "
      f"(paper: 15.5%)")
