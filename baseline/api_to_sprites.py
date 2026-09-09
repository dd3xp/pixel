"""Turn the API-generated 1024 px renders into 16 px RGBA sprites through the project's OWN preprocessing.

Runs on the server so it can import src/v6/train_v7.to_tensor: the same BOX downscale, the same hard alpha
threshold and the same centred canvas our own samples go through, so the FD comparison is on identical footing.
"""
import argparse, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src/v6")
from train_v7 import to_tensor


def to_rgba(x):
    a = ((x.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a, "RGBA")


def cutout(img, tol=18):
    from collections import deque
    rgb = np.asarray(img.convert("RGB"))
    a = rgb.astype(np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(a - bg).max(axis=2) <= tol
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True; q.append((ny, nx))
    alpha = (~seen).astype(np.uint8) * 255
    out = np.dstack([rgb, alpha])
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return Image.fromarray(out, "RGBA"), 1.0
    frac = float((alpha > 0).mean())
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA"), frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=16)
    args = ap.parse_args()
    outdir = Path(args.out) / f"s{args.size}"
    outdir.mkdir(parents=True, exist_ok=True)
    fs = sorted(Path(args.big).glob("*.png")) + sorted(Path(args.big).glob("*.jpg"))
    fracs = []
    for f in fs:
        im = Image.open(f)
        sprite, frac = cutout(im)
        fracs.append(frac)
        x = to_tensor(sprite, args.size)
        to_rgba(x).save(outdir / (f.stem + ".png"))
    fr = np.array(fracs)
    print(f"wrote {len(fs)} sprites -> {outdir}")
    print(f"cutout kept fraction: median {np.median(fr):.3f}, >0.98 (cutout failed) on {(fr > 0.98).mean():.1%} of images")


if __name__ == "__main__":
    main()
