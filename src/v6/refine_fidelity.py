"""How much of the external system's sprite survives the refinement?

refine_ext.sh re-noises an external baseline's 16 px sprite and denoises it with our model under the
reported configuration.  Native FD improves a lot, but the higher the strength the less of the input is
left, and at strength 1.0 the result is simply our own model.  A distributional score alone therefore
cannot separate "we repaired their sprite" from "we replaced it".  This reports similarity to the input
with our own samples for the same prompts as the floor.  Note that the DINOv2 cosine is useless here --
two unrelated sprites for the same prompt already score 0.918 at 16 px -- so the pixel-space measures
are the ones to read: alpha-mask IoU and mean RGB distance over the shared opaque pixels.

Usage: python src/v6/refine_fidelity.py --runs runs_out/refine/refine_gpt_s0p4 ... --init <dir> --ours <dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fd_dino as FD  # noqa: E402


def by_index(d):
    import re
    out = {}
    for f in Path(d).glob("*.png"):
        m = re.match(r"0*(\d+)", f.stem)
        if m:
            out.setdefault(int(m.group(1)), f)
    return out


def cosine_to(ref_dir, gen_dir, R):
    a, b = by_index(ref_dir), by_index(gen_dir)
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan"), 0
    fa = FD.embed([str(a[k]) for k in keys], R)
    fb = FD.embed([str(b[k]) for k in keys], R)
    fa = fa / np.linalg.norm(fa, axis=1, keepdims=True)
    fb = fb / np.linalg.norm(fb, axis=1, keepdims=True)
    return float((fa * fb).sum(1).mean()), len(keys)


def pixel_stats(ref_dir, gen_dir):
    """Alpha-mask IoU and mean RGB L1 (0-255) over pixels opaque in both, on the sprite canvas."""
    from PIL import Image
    a, b = by_index(ref_dir), by_index(gen_dir)
    keys = sorted(set(a) & set(b))
    ious, l1s = [], []
    for k in keys:
        x = np.array(Image.open(a[k]).convert("RGBA")).astype(np.float32)
        y = np.array(Image.open(b[k]).convert("RGBA")).astype(np.float32)
        if x.shape != y.shape:
            continue
        mx, my = x[:, :, 3] >= 128, y[:, :, 3] >= 128
        u = (mx | my).sum()
        ious.append(((mx & my).sum() / u) if u else 1.0)
        both = mx & my
        if both.any():
            l1s.append(np.abs(x[:, :, :3][both] - y[:, :, :3][both]).mean())
    iou = float(np.mean(ious)) if ious else float("nan")
    l1 = float(np.mean(l1s)) if l1s else float("nan")
    return iou, l1, len(ious)


def report(init, d, label, size):
    c, n = cosine_to(init, d, size)
    iou, l1, m = pixel_stats(init, d)
    print(f"{label:34s} IoU {iou:.3f}  RGB-L1 {l1:6.1f}  (DINO cos {c:.3f}, n={m})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="the external system's sprites, i.e. the input")
    ap.add_argument("--runs", nargs="+", required=True, help="refined output directories")
    ap.add_argument("--ours", default=None, help="our own samples for the same prompts: the cosine floor")
    ap.add_argument("--size", type=int, default=16)
    a = ap.parse_args()
    if a.ours:
        report(a.init, a.ours, "ours, no init (floor)", a.size)
    for r in a.runs:
        report(a.init, r, Path(r).parent.name, a.size)


if __name__ == "__main__":
    main()
