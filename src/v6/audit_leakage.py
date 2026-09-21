"""Recompute the duplicate and leakage figures the abstract rests on.

Claims under test:
  * 4,888 pixel-identical duplicate copies inside the corpus;
  * a twin in training for 15.5% of the held-out set;
  * 41.0% of the native 16 px held-out prompts have a near-duplicate twin.
These are data facts rather than self-evaluation, and the data facts have held up so far, but the
abstract is the worst place to carry a wrong number.  Scope matters and the first version of this
script got it wrong three times over: the duplicate count is over the whole training corpus
(oga_clean + extra_all), not oga_clean alone, and the near-duplicate figure compares the held-out
prompts against the *training* sprites, not against the reference half.

Usage: python src/v6/audit_leakage.py
"""
import glob
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fd_fair import real_split  # noqa: E402


def pixhash(p):
    a = np.array(Image.open(p).convert("RGBA"))
    return hashlib.md5(a.tobytes()).hexdigest() + f"_{a.shape[0]}x{a.shape[1]}"


def ahash(p):
    im = Image.open(p).convert("RGBA").resize((8, 8), Image.BOX)
    a = np.array(im).astype(np.float32)
    lum = a[:, :, :3].mean(2) * (a[:, :, 3] / 255.0)
    return (lum > lum.mean()).tobytes() + (a[:, :, 3] > 127).tobytes()


def main():
    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    allc = sorted(set(real) | {p.replace("\\", "/") for p in
                               glob.glob("data/extra_all/**/*.png", recursive=True)
                               + glob.glob("data/extra_all/*.png")})
    print(f"oga_clean: {len(real)} unique; whole training corpus: {len(allc)}", flush=True)

    groups = defaultdict(list)
    for p in allc:
        try:
            groups[pixhash(p)].append(p)
        except Exception:
            continue
    extra = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    print(f"pixel-identical duplicate copies (paper: 4,888): {extra}", flush=True)

    # hold-out leakage: a held-out sprite whose pixel hash also appears among the training sprites
    ref, held = real_split(native_R=None, pool="old")
    held = [p.replace("\\", "/") for p in held][:3000]
    heldset = set(held)
    train = [p for p in real if p not in heldset]
    trainh = Counter()
    for p in train:
        try:
            trainh[pixhash(p)] += 1
        except Exception:
            continue
    twin = 0
    for p in held:
        try:
            if trainh.get(pixhash(p), 0) > 0:
                twin += 1
        except Exception:
            continue
    print(f"held-out sprites with an exact twin in training (paper: 15.5%): {100 * twin / len(held):.1f}%",
          flush=True)

    # near-duplicate twins among the native 16 px held-out set
    ref16, held16 = real_split(native_R=16, pool="old")
    # the published 41.0% compares the held-out prompts against the TRAINING sprites
    train16 = [p for p in allc if p not in set(held16)]
    tr = {}
    for p in train16:
        try:
            tr[ahash(p)] = 1
        except Exception:
            continue
    n_twin = 0
    for p in held16:
        try:
            if ahash(p) in tr:
                n_twin += 1
        except Exception:
            continue
    print(f"native 16px held-out with a near-duplicate in the reference half (paper: 41.0%): "
          f"{100 * n_twin / max(1, len(held16)):.1f}%  (n={len(held16)})", flush=True)


if __name__ == "__main__":
    main()
