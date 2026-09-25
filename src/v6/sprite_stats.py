"""Pixel statistics of a sprite at a target resolution, without importing the training stack.

train_cond.to_tensor is the authority on how a file becomes a 16 px sprite, but importing it drags in
torch and diffusers, which the analysis box does not have. This reproduces the same two steps --
premultiplied BOX downscale for oversize art, integer NEAREST upscale for art at most half the bucket,
then centre on a transparent canvas with alpha hard-thresholded at 128 -- and returns PIL images, so
the numbers here are the numbers the pipeline would see.

The three statistics are the ones the paper uses to describe the medium:

  coverage  fraction of the canvas that is opaque
  colours   distinct opaque RGB values
  flat      fraction of adjacent opaque pairs that are exactly equal, i.e. how much of the sprite is
            flat colour rather than gradient. Real sprites drawn at 16 px average 0.46; art shrunk
            into the bucket is far lower, because BOX averaging invents a new colour per pixel.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE_CACHE = Path("runs_out/native_sizes.json")


def native_sizes(paths):
    """max(side) per file, sharing fd_fair's cache so both agree on what 'native' means."""
    d = json.load(open(SIZE_CACHE)) if SIZE_CACHE.exists() else {}
    miss = [p for p in paths if p not in d]
    for p in miss:
        d[p] = max(Image.open(p).size)
    if miss:
        SIZE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(d, open(SIZE_CACHE, "w"))
    return d


def downscale_rgba(im, side):
    f = side / max(im.size)
    w, h = max(1, round(im.width * f)), max(1, round(im.height * f))
    a = np.array(im).astype(np.float32)
    a[:, :, :3] *= a[:, :, 3:4] / 255.0
    pm = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA").resize((w, h), Image.BOX)
    b = np.array(pm).astype(np.float32)
    alpha = b[:, :, 3:4]
    b[:, :, :3] = np.where(alpha > 0, b[:, :, :3] / np.maximum(alpha, 1) * 255.0, 0)
    return Image.fromarray(b.clip(0, 255).astype(np.uint8), "RGBA")


def at_size(path, side):
    im = Image.open(path).convert("RGBA")
    if max(im.size) > side:
        im = downscale_rgba(im, side)
    elif max(im.size) * 2 <= side:
        f = side // max(im.size)
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.array(canvas).astype(np.float32)
    a[a[:, :, 3] < 128] = 0.0
    a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def stats(im):
    a = np.array(im)
    op = a[:, :, 3] > 127
    n = int(op.sum())
    if n == 0:
        return {"coverage": 0.0, "colours": 0, "flat": 0.0, "opaque": 0}
    rgb = a[:, :, :3]
    cols = {tuple(v) for v in rgb[op]}
    pairs = eq = 0
    for axis in (0, 1):
        both = op & np.roll(op, -1, axis=axis)
        both = both[:-1, :] if axis == 0 else both[:, :-1]
        same = (rgb == np.roll(rgb, -1, axis=axis)).all(-1)
        same = same[:-1, :] if axis == 0 else same[:, :-1]
        pairs += int(both.sum())
        eq += int((both & same).sum())
    return {"coverage": n / a.shape[0] / a.shape[1], "colours": len(cols),
            "flat": eq / max(1, pairs), "opaque": n}


def real_split(native_R=None):
    """fd_fair.real_split without the diffusers import, byte-for-byte the same split.

    Reproduced here because fd_fair imports train_cond for to_tensor, and train_cond imports diffusers,
    which the shared environment can no longer load. The seeds, the double glob, the dedup and the
    half-split are copied exactly, so the held-out half this returns is the one the probe must exclude.
    """
    import glob as _glob
    import random as _random
    real = _glob.glob("data/oga_clean/**/*.png", recursive=True) + _glob.glob("data/oga_clean/*.png")
    if native_R:
        real = sorted({p.replace("\\", "/") for p in real})
        sz = native_sizes(real)
        real = [p for p in real if sz[p] <= native_R]
    _random.seed(0)
    _random.shuffle(real)
    ref = real[:len(real) // 2] if native_R else real[:3000]
    refset = {p.replace("\\", "/") for p in ref}
    held = sorted({p.replace("\\", "/") for p in real} - refset)
    _random.seed(2)
    _random.shuffle(held)
    return ref, held
