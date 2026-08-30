"""Make the pseudo-labels look like pixel art before training on them.

v11 gained vocabulary (objects that were unrenderable became recognisable) but
paid for it: in-domain sprites came back washed out.  Measuring the two corpora
explains exactly why -- the pseudo-labels violate both defining properties of
the medium:

                saturation   brightness   distinct colours
    real            106.5        133.9          59.3
    pseudo           27.4        159.4         134.0
    ratio           x0.26        x1.19          x2.26

A quarter of the saturation and more than twice the palette: downscaling an SDXL
render produces anti-aliased gradients, which is the opposite of a hand-picked
limited palette.  Training on that teaches the model a distribution that fights
the one the real sprites teach.

Two corrections, each aimed at one measured gap:
  * adaptive median-cut palette per sprite -- NOT the fixed DawnBringer32 used
    in the downscale baseline, which hurt there precisely because forcing
    arbitrary images into one fixed palette produces speckle.  Choosing the
    palette from the sprite's own colours keeps its identity.
  * saturation scaled toward the real corpus mean, with brightness pulled back.

Alpha is kept binary: a sprite either covers a pixel or it does not, and soft
edges are another thing real sprites do not have.

Usage: python src/v6/fix_pseudo.py --src data/pseudo --dst data/pseudo_fix
"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
TARGET_SAT = 106.5          # measured mean over 400 real sprites
TARGET_COLOURS = 48
# The gate has to sit between "truly achromatic" and "coloured but washed".
# At 60 almost every pseudo-label pixel counted as grey (their mean saturation
# is only 27) and the correction did nothing; at 25 a washed-out colour gets the
# full boost while a grey camera or a black-and-white football stays neutral.
SAT_GATE = 25
MAX_BOOST = 2.5      # a flat ~4x turned greys into false colour


def fix(path, out_path):
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    alpha = (a[:, :, 3] > 128).astype(np.uint8) * 255      # binary coverage
    if alpha.sum() == 0:
        return False

    rgb = Image.fromarray(a[:, :, :3], "RGB")
    hsv = np.array(rgb.convert("HSV")).astype(np.float32)
    m = alpha > 0
    # Boost saturation in proportion to how colourful a pixel already is.
    # A flat multiplier looked right on the average but wrecked achromatic
    # objects: at ~4x, the faint colour noise in a grey camera or a black-and
    # -white football became strong false colour (football turned purple,
    # trumpet turned blue).  Gating on the pixel's own saturation leaves greys
    # grey and only deepens colours that were really there.
    sat = hsv[:, :, 1]
    colourful = sat[m & (sat > SAT_GATE)]
    if colourful.size > 16:
        k = min(MAX_BOOST, TARGET_SAT / max(colourful.mean(), 1.0))
        gate = np.clip(sat / SAT_GATE, 0, 1)          # 0 for grey, 1 once colourful
        hsv[:, :, 1] = np.clip(sat * (1 + (k - 1) * gate), 0, 255)
    # the pseudo-labels also run bright; ease that back so the boosted
    # saturation does not just read as neon
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.92, 0, 255)
    rgb = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")

    # adaptive palette from this sprite's own colours
    q = rgb.quantize(colors=TARGET_COLOURS, method=Image.MEDIANCUT).convert("RGB")
    out = np.dstack([np.array(q), alpha])
    Image.fromarray(out, "RGBA").save(out_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pseudo")
    ap.add_argument("--dst", default="data/pseudo_fix")
    args = ap.parse_args()
    src, dst = ROOT / args.src, ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in sorted(src.glob("*.png")):
        if fix(p, dst / p.name):
            n += 1
    shutil.copyfile(ROOT / f"{args.src}.csv", ROOT / f"{args.dst}.csv")
    print(f"{n} sprites -> {args.dst}")

    def stats(paths):
        S, C = [], []
        for p in paths:
            a = np.array(Image.open(p).convert("RGBA"))
            m = a[:, :, 3] > 128
            if m.sum() < 4:
                continue
            hsv = np.array(Image.fromarray(a[:, :, :3]).convert("HSV"))[m]
            S.append(hsv[:, 1].mean())
            C.append(len({tuple(x) for x in a[:, :, :3][m]}))
        return float(np.mean(S)), float(np.mean(C))

    s, c = stats(sorted(dst.glob("*.png"))[:400])
    print(f"after fix: saturation {s:.1f} (real 106.5), colours {c:.1f} (real 59.3)")


if __name__ == "__main__":
    main()
