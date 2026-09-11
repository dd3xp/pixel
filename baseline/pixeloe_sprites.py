"""External baseline: a big text-to-image model's 1024 px render + PROFESSIONAL pixelization (PixelOE),
instead of the naive BOX downscale used so far.

Why it exists: the survey (external_baselines_survey.md) flagged that comparing only against naive downscaling
invites the reviewer objection "you just shrank the image badly". PixelOE is a widely used contrast-aware
pixelizer (outline expansion + k-centroid/contrast downscale + palette matching), so this is the strongest
honest version of "generate big, then make it pixel art".

PixelOE has no alpha, so: cut the sprite out of its plain background first (same border flood-fill as the other
external baselines), pixelize only the RGB, shrink the alpha mask to the same grid by majority vote, recombine,
and finally centre it with the project's own canvas logic. Scored with the unchanged fd_fair / fd_dino.

Usage: python baseline/pixeloe_sprites.py --big runs/ext_recap_gpt/big --out runs/ext_recap_gpt_pixeloe --size 16
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "baseline")
from api_gen import cutout
from pixeloe.legacy.pixelize import pixelize


def to_grid(rgba, size, mode):
    rgb = np.asarray(rgba.convert("RGB"))
    alpha = np.asarray(rgba)[..., 3]
    h, w = alpha.shape
    side = max(h, w)
    # pad to a square so PixelOE's grid maps 1:1 onto the target canvas
    sq_rgb = np.full((side, side, 3), 255, np.uint8)
    sq_a = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    sq_rgb[y0:y0 + h, x0:x0 + w] = rgb
    sq_a[y0:y0 + h, x0:x0 + w] = alpha
    patch = max(1, side // size)
    out = pixelize(sq_rgb, mode=mode, target_size=size, patch_size=patch, no_upscale=True, color_matching=True)
    out = np.asarray(Image.fromarray(np.asarray(out)).convert("RGB").resize((size, size), Image.NEAREST))
    # alpha by majority vote inside each cell
    a_small = np.asarray(Image.fromarray(sq_a).resize((size, size), Image.BOX))
    a_small = (a_small >= 128).astype(np.uint8) * 255
    rgba_small = np.dstack([out, a_small])
    rgba_small[a_small == 0] = 0
    return Image.fromarray(rgba_small, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--mode", default="contrast", choices=["contrast", "k-centroid", "center", "nearest", "bicubic"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out) / f"s{a.size}"
    out.mkdir(parents=True, exist_ok=True)
    fs = sorted(Path(a.big).glob("*.png"))
    if a.limit:
        fs = fs[:a.limit]
    n = 0
    for f in fs:
        try:
            sprite = cutout(Image.open(f))
            to_grid(sprite, a.size, a.mode).save(out / f.name)
            n += 1
        except Exception as e:
            print(f"  skip {f.name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    print(f"wrote {n} PixelOE sprites -> {out}")


if __name__ == "__main__":
    main()
