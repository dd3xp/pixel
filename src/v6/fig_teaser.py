"""Teaser: what a 16 px sprite is, what the corpus calls one, and what generators produce.

Four rows of eight, all rendered at 16 px on the same checkerboard:
  A  real sprites that are natively 16 px or smaller -- the medium: a few flat colours, hard edges;
  B  sprites the corpus feeds into the same 16 px bucket, which are 25-64 px artwork box-downscaled --
     soft, dozens of colours, and 92.9% of the conventional evaluation reference;
  C  gpt-image-2 rendered large and reduced -- what a practitioner actually gets, without the palette
     post-process the external table applies for fairness, because the point here is the colour count;
  D  ours.
Rows A and B are unmatched samples of two populations; rows C and D share the caption of the column.
Each label carries the median number of opaque colours over the row, which is the point of the figure:
the eye reads row D as rougher than row C, and the colour count says row D is the one in the medium.

Usage: python src/v6/fig_teaser.py --out runs_out/fig_teaser.png
"""
import argparse
import glob
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from fd_fair import native_sizes  # noqa: E402
from train_cond import to_tensor  # noqa: E402

CHECK = ((235, 235, 235), (205, 205, 205))


def board(im, scale, cellpx):
    im = im.convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 4) + (y // 4)) % 2]
    bg.paste(im, (0, 0), im)
    return bg.resize((cellpx, cellpx), Image.NEAREST) if bg.size != (cellpx, cellpx) else bg


def at_R(path, R):
    t = to_tensor(Image.open(path).convert("RGBA"), R)
    a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a, "RGBA")


def by_index(d):
    out = {}
    for f in Path(d).glob("*.png"):
        m = re.match(r"0*(\d+)", f.stem)
        if m:
            out.setdefault(int(m.group(1)), f)
    return out


def median_colours(ims):
    import numpy as np
    vals = []
    for im in ims:
        a = np.array(im.convert("RGBA"))
        vis = a[a[:, :, 3] >= 128][:, :3]
        if len(vis):
            vals.append(len(np.unique(vis.reshape(-1, 3), axis=0)))
    return int(np.median(vals)) if vals else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_out/fig_teaser.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = a.size
    rng = random.Random(a.seed)

    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    native = [p for p in real if sz[p] <= R]
    downscaled = [p for p in real if 25 <= sz[p] <= 64]
    print(f"{len(native)} natively <= {R}px, {len(downscaled)} at 25-64px")

    gpt = by_index(f"runs_out/ext200/runs/ext_recap_gpt/s{R}")
    ours = by_index(f"runs_out/ext200/v8n/icg2_pal4/s{R}")
    cols = sorted(rng.sample(sorted(set(gpt) & set(ours)), a.n))

    rows = [
        ("real, natively 16 px", [at_R(p, R) for p in rng.sample(native, a.n)]),
        ("corpus 16 px targets", [at_R(p, R) for p in rng.sample(downscaled, a.n)]),
        ("gpt-image-2, reduced", [Image.open(gpt[c]) for c in cols]),
        ("ours", [Image.open(ours[c]) for c in cols]),
    ]

    # 16x instead of 8x: at CVPR two-column width the 8x sheet is about 170 dpi, which
    # prints visibly soft even though the art is nearest-neighbour and lossless
    scale, cellpx, pad, lab = 16, 256, 8, 256
    W = lab + a.n * (cellpx + pad)
    H = len(rows) * (cellpx + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for r, (name, ims) in enumerate(rows):
        dr.text((4, r * (cellpx + pad) + cellpx // 2 - 4), name, fill=(0, 0, 0))
        med = median_colours(ims)
        dr.text((4, r * (cellpx + pad) + cellpx // 2 + 8), f"median {med} colours", fill=(90, 90, 90))
        for c, im in enumerate(ims):
            sheet.paste(board(im, scale, cellpx), (lab + c * (cellpx + pad), r * (cellpx + pad)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out} ({W}x{H})")


if __name__ == "__main__":
    main()
