"""Simplicity statistics of sprite sets: is the model's lower-bucket belief a *simpler sprite* (fewer colours,
larger flat regions) or a *low-pass* image (blended colours)?  Mechanism check for cross-resolution
autoguidance (experiment_log 09-07): the explicit block-average branch (probe_cg) failed, the bucket:12 label
works, so the two should differ in exactly these statistics.

Per image (opaque = alpha >= 128): opaque fraction, #unique RGB colours, flat ratio (fraction of opaque-opaque
4-neighbour pairs with identical RGB), mean |dRGB| over those pairs (TV, 0-255 scale).  Reports median / mean.

Usage:
  python src/v6/stats_simplicity.py --dirs belief12=runs_out/v7h_bk12only_matched_eval/s16 v7h=runs_out/v7h_matched_eval/s16 \
      real16=runs_out/ref_totensor_s16 --real_bucket 12 16 24 --block real16:2
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_cond import to_tensor  # noqa: E402

BUCKETS = [12, 16, 20, 24, 32, 48, 64]


def stats(arr):
    """arr: uint8 HxWx4."""
    a = arr[..., 3] >= 128
    if a.sum() < 2:
        return None
    rgb = arr[..., :3].astype(np.int16)
    ncol = len(np.unique(rgb[a].reshape(-1, 3), axis=0))
    pairs_same = pairs = 0
    tv = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        m = a[dy:, dx:] & a[: a.shape[0] - dy, : a.shape[1] - dx]
        d = np.abs(rgb[dy:, dx:] - rgb[: a.shape[0] - dy, : a.shape[1] - dx]).sum(-1)[m]
        pairs += m.sum()
        pairs_same += (d == 0).sum()
        tv += d.sum()
    if pairs == 0:
        return None
    return dict(opaque=a.mean(), ncol=ncol, flat=pairs_same / pairs, tv=tv / pairs / 3)


def load_dir(d, block=None):
    out = []
    for f in sorted(glob.glob(f"{d}/*.png")):
        arr = np.asarray(Image.open(f).convert("RGBA"))
        if block:
            arr = block_avg(arr, block)
        out.append(arr)
    return out


def block_avg(arr, k):
    """k x k block average of RGBA (alpha averaged too, then thresholded), nearest-upsampled back, like train_coarse.degrade."""
    h, w = arr.shape[:2]
    x = arr.astype(np.float32)
    ph, pw = (-h) % k, (-w) % k
    x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="edge")
    x = x.reshape((h + ph) // k, k, (w + pw) // k, k, 4).mean((1, 3))
    x = np.repeat(np.repeat(x, k, 0), k, 1)[:h, :w]
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def real_bucket(b, side, n=3000):
    """Native sprites whose bucket (smallest B >= max side) is b, pasted onto a `side` canvas via to_tensor."""
    files = sorted(glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png"))
    out = []
    for f in files:
        im = Image.open(f).convert("RGBA")
        bk = next((B for B in BUCKETS if B >= max(im.size)), None)
        if bk != b:
            continue
        x = to_tensor(im, side)
        out.append(np.clip(np.round((x.permute(1, 2, 0).numpy() + 1) * 127.5), 0, 255).astype(np.uint8))
        if len(out) >= n:
            break
    return out


def report(name, arrs):
    rows = [s for s in (stats(a) for a in arrs) if s is not None]
    if not rows:
        print(f"{name:<28} (empty)")
        return
    def agg(k):
        v = np.array([r[k] for r in rows])
        return f"{np.median(v):7.3f}/{v.mean():7.3f}"
    print(f"{name:<28} n={len(rows):5d}  opaque {agg('opaque')}  ncol {agg('ncol')}  flat {agg('flat')}  tv {agg('tv')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[], help="name=dir of RGBA pngs")
    ap.add_argument("--real_bucket", type=int, nargs="*", default=[], help="native buckets to report as reals")
    ap.add_argument("--side", type=int, default=16, help="canvas side for --real_bucket")
    ap.add_argument("--block", nargs="*", default=[], help="name:k -> also report k x k block-averaged version of that dir")
    args = ap.parse_args()
    print(f"{'set':<28} {'':8} {'median/mean':>12}")
    dirs = dict(x.split("=", 1) for x in args.dirs)
    for name, d in dirs.items():
        report(name, load_dir(d))
    for spec in args.block:
        name, k = spec.split(":")
        report(f"{name}_block{k}", load_dir(dirs[name], int(k)))
    for b in args.real_bucket:
        report(f"real_native{b}@{args.side}", real_bucket(b, args.side))


if __name__ == "__main__":
    main()
