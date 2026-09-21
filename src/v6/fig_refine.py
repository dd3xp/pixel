"""Figure for the post-process result: an external sprite, the same sprite refined at two strengths,
our own sample for the caption, and the held-out sprite.

One column per caption, drawn at random from the 200-prompt external set with a fixed seed.  An earlier
version ranked captions by how much the refinement moved them, which selects for the cases where the
subject is lost and makes the method look worse than the aggregate statistics say it is; --pick changed
keeps that view for inspection, but the figure in the paper is the random one.

Usage: python src/v6/fig_refine.py --sys gpt --n 8 --out runs_out/fig_refine.png
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CHECK = ((235, 235, 235), (205, 205, 205))


def by_index(d):
    out = {}
    for f in Path(d).glob("*.png"):
        m = re.match(r"0*(\d+)", f.stem)
        if m:
            out.setdefault(int(m.group(1)), f)
    return out


def cell(path, scale, cellpx):
    im = Image.open(path).convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 4) + (y // 4)) % 2]
    bg.paste(im, (0, 0), im)
    if bg.size != (cellpx, cellpx):
        bg = bg.resize((cellpx, cellpx), Image.NEAREST)
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", default="gpt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs_out/fig_refine.png")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--pick", default="random", choices=["random", "changed"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [
        (f"{a.sys}", f"runs_out/ext200/q/{a.sys}_q4/s16"),
        ("refined 0.4", f"runs_out/refine/refine_{a.sys}_s0p4/s16"),
        ("refined 0.6", f"runs_out/refine/refine_{a.sys}_s0p6/s16"),
        ("ours", "runs_out/ext200/v8n/icg2_pal4/s16"),
        ("real", "runs_out/ext200/q/real_q4/s16"),
    ]
    idx = [by_index(d) for _, d in rows]
    keys = sorted(set.intersection(*(set(i) for i in idx)))
    print(f"{len(keys)} captions common to all five rows")

    if a.pick == "random":
        import random
        picked = sorted(random.Random(a.seed).sample(keys, min(a.n, len(keys))))
    else:
        src, ref = idx[0], idx[2]
        score = {}
        for k in keys:
            x = np.array(Image.open(src[k]).convert("RGBA")).astype(np.float32)
            y = np.array(Image.open(ref[k]).convert("RGBA")).astype(np.float32)
            if x.shape == y.shape:
                score[k] = float(np.abs(x - y).mean())
        picked = sorted(sorted(score, key=score.get, reverse=True)[:a.n])

    scale, cellpx, pad, lab = 8, 128, 3, 78
    W = lab + len(picked) * (cellpx + pad)
    H = len(rows) * (cellpx + pad)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for r, ((name, _), m) in enumerate(zip(rows, idx)):
        dr.text((4, r * (cellpx + pad) + cellpx // 2 - 4), name, fill=(0, 0, 0))
        for c, k in enumerate(picked):
            sheet.paste(cell(m[k], scale, cellpx), (lab + c * (cellpx + pad), r * (cellpx + pad)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print(f"wrote {a.out} ({W}x{H}), captions {picked}")


if __name__ == "__main__":
    main()
