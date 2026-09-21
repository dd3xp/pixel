"""Which definition of "flat" produced the 0.36-0.38 in the method section?

stats_simplicity.py counts 4-neighbour pairs with *identical* RGB and gives 0.083 for plain CFG, while
the paragraph in the paper quotes 0.36-0.38 for the same runs.  One of the two is wrong.  This prints
the flat ratio under exact equality and under a few tolerances, so the definition that reproduces the
published number can be identified before either is changed.
"""
import glob
import sys

import numpy as np
from PIL import Image


def flat(path, tol):
    a = np.array(Image.open(path).convert("RGBA")).astype(np.int16)
    op = a[:, :, 3] >= 128
    rgb = a[:, :, :3]
    num = den = 0
    for s, s2 in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                  ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        m = op[s] & op[s2]
        if not m.any():
            continue
        d = np.abs(rgb[s] - rgb[s2]).sum(-1)[m]
        num += int((d <= tol).sum())
        den += int(m.sum())
    return num / den if den else np.nan


for d in sys.argv[1:]:
    files = sorted(glob.glob(d + "/*.png"))[:600]
    if not files:
        print(f"{d}: empty")
        continue
    row = []
    for tol in (0, 4, 8, 16, 32):
        vals = [flat(f, tol) for f in files]
        row.append(f"tol{tol}={np.nanmean(vals):.3f}")
    print(f"{d.split('/')[-2] if d.endswith('s16') else d:42s} n={len(files):4d}  " + "  ".join(row))
