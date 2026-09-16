"""Second pass over the rebuilt corpus: drop non-sprite sheets and collapse animation frames.

The pixel-hash dedup in build_corpus_v8.py only catches *identical* copies.  Two problems survive it:
  * 1,263 of 22,980 sprites come from font / UI / text sheets, which are not objects and whose glyphs
    slipped past the colour filters;
  * a single animation sheet contributes up to 563 near-identical frames (warpsara-shooting-sheet 269,
    blue_alien 239), which would let one character dominate a 16 px bucket.
Frames are collapsed with an average hash over a 8x8 luma + alpha grid (near-duplicates share a hash),
then each source sheet is capped so no sheet can flood a bucket.

Writes data/oga3_keep.txt (one kept filename per line) and prints the audit.
Usage: python src/v6/filter_corpus_v8.py [--cap 60]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

JUNK = re.compile(r"font|glyph|letter|alphabet|numbers?[-_.]|\bui\b|gui|button|menu|cursor|"
                  r"text|logo|title|bar[-_.]|hud", re.I)


def ahash(p):
    im = Image.open(p).convert("RGBA").resize((8, 8), Image.BOX)
    a = np.array(im).astype(np.float32)
    lum = (a[:, :, :3].mean(2) * (a[:, :, 3] / 255.0))
    return (lum > lum.mean()).tobytes() + (a[:, :, 3] > 127).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60, help="max sprites kept from one source sheet")
    a = ap.parse_args()
    rows = list(csv.DictReader(open("data/oga3_manifest.csv", encoding="utf-8")))
    by_sheet = defaultdict(list)
    n_junk = 0
    for r in rows:
        if JUNK.search(r["member"]):
            n_junk += 1
            continue
        by_sheet[(r["src"], r["zip"], r["member"])].append(r)

    keep, n_frame = [], 0
    rng = np.random.default_rng(0)
    for sheet, items in by_sheet.items():
        seen = {}
        for r in items:
            try:
                h = ahash(Path("data/oga3_clean") / r["path"])
            except Exception:
                continue
            if h in seen:
                n_frame += 1
                continue
            seen[h] = r
        uniq = list(seen.values())
        if len(uniq) > a.cap:
            idx = rng.choice(len(uniq), a.cap, replace=False)
            uniq = [uniq[i] for i in sorted(idx)]
        keep.extend(uniq)

    sides = np.array([int(r["side"]) for r in keep])
    pals = np.array([int(r["palette"]) for r in keep])
    Path("data/oga3_keep.txt").write_text("\n".join(r["path"] for r in keep), encoding="utf-8")
    print(f"manifest {len(rows)} -> junk sheets {n_junk}, near-duplicate frames {n_frame}, "
          f"sheet cap {a.cap} -> kept {len(keep)} from {len(by_sheet)} sheets", flush=True)
    for R in (12, 16, 20, 24, 32, 64):
        print(f"  native <= {R:2d}px: {(sides <= R).sum()}")
    print(f"  palette median {np.median(pals):.0f}, p90 {np.percentile(pals, 90):.0f}")


if __name__ == "__main__":
    main()
