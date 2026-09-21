"""Recompute every corpus statistic the paper asserts, from the data, in one place.

Two claims have already turned out to be wrong when checked (the palette-size "sanity check" and the
flat-neighbour definition), both written from remembered numbers rather than from a rerun.  This script
exists so the remaining ones are not taken on trust either.  It prints the paper's value next to the
recomputed one and marks disagreements.

Usage: python src/v6/audit_corpus_claims.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fd_fair import native_sizes, real_split  # noqa: E402

CLAIMS = {
    "native <= 16 px share of the corpus": 7.1,
    "biased 16 px reference that is downscaled artwork": 92.9,
    "real 16 px sprites using at most four colours": 22.6,
    "median opaque colours, real native 16 px": 6,
    "median opaque colours, real native 12 px": 5,
    "median opaque colours, real native 20 px": 7,
    "median opaque colours, real native 24 px": 7,
}


def palette(p, R):
    from train_cond import to_tensor
    t = to_tensor(Image.open(p).convert("RGBA"), R)
    a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    vis = a[a[:, :, 3] >= 128][:, :3]
    return len(np.unique(vis.reshape(-1, 3), axis=0)) if len(vis) else 0


def line(name, got, fmt="{:.1f}"):
    want = CLAIMS.get(name)
    ok = "" if want is None else ("  OK" if abs(got - want) <= (0.15 if want > 1 else 0.05) * max(1, abs(want)) * 0.5 + 0.35
                                  else f"  <-- paper says {want}")
    print(f"{name:52s} {fmt.format(got)}{ok}", flush=True)


def main():
    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    n = len(real)
    print(f"corpus (data/oga_clean, unique): {n} sprites", flush=True)
    line("native <= 16 px share of the corpus", 100 * sum(1 for p in real if sz[p] <= 16) / n)

    # the biased reference is a random 3,000 of the corpus; how much of it is not natively small?
    ref, _ = real_split(native_R=None, pool="old")
    ref = [p.replace("\\", "/") for p in ref]
    big = sum(1 for p in ref if sz.get(p, 999) > 16)
    line("biased 16 px reference that is downscaled artwork", 100 * big / len(ref))

    import random
    for R in (12, 16, 20, 24):
        pool = [p for p in real if sz[p] <= R]
        # a sorted prefix is an alphabetical subset, i.e. a few source sheets; sample instead
        take = pool if len(pool) <= 3000 else random.Random(0).sample(pool, 3000)
        print(f"  ({R} px pool {len(pool)}, measured on {len(take)})", flush=True)
        pals = [palette(p, R) for p in take]
        line(f"median opaque colours, real native {R} px", float(np.median(pals)), "{:.0f}")
        if R == 16:
            line("real 16 px sprites using at most four colours",
                 100 * float(np.mean([p <= 4 for p in pals])))


if __name__ == "__main__":
    main()
