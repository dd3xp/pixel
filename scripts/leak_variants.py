"""Which definition of "leaked" gives the published 15.5%?

Three reconstructions have missed it (14.6%, 11.7%, ambiguous).  Rather than guess again one at a time,
this computes every reasonable definition on the same split and prints them together, so the published
figure can be matched to a definition or declared unreproducible with evidence.

Split: fd_fair.real_split() with no native filter, held[:3000] -- the set heldout3000_* was built from.
"""
import glob
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src/v6")
from fd_fair import real_split  # noqa: E402


def pixhash(p):
    a = np.array(Image.open(p).convert("RGBA"))
    return hashlib.md5(a.tobytes()).hexdigest() + f"_{a.shape[0]}x{a.shape[1]}"


oga = {p.replace("\\", "/") for p in
       glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")}
extra = {p.replace("\\", "/") for p in
         glob.glob("data/extra_all/**/*.png", recursive=True) + glob.glob("data/extra_all/*.png")}
allc = sorted(oga | extra)

ref, held = real_split(native_R=None, pool="old")
held = [p.replace("\\", "/") for p in held][:3000]
heldset = set(held)

H = {}
for p in allc:
    try:
        H[p] = pixhash(p)
    except Exception:
        pass
cnt_all = Counter(H[p] for p in allc if p in H)
cnt_oga = Counter(H[p] for p in oga if p in H)

defs = {
    "twin anywhere in the whole corpus (oga+extra)":
        sum(1 for p in held if p in H and cnt_all[H[p]] > 1),
    "twin in oga_clean only":
        sum(1 for p in held if p in H and cnt_oga[H[p]] > 1),
    "twin outside the held-out set (oga+extra)":
        sum(1 for p in held if p in H and cnt_all[H[p]] - sum(
            1 for q in heldset if q in H and H[q] == H[p]) > 0),
}
for name, n in defs.items():
    print(f"{name:52s} {n:5d}/3000 = {100 * n / 3000:.1f}%", flush=True)
print("paper: 15.5%", flush=True)
