"""Before training on it: does the margin policy actually move the 16 px bucket's coverage to 26%?

The standing rule is that no new target distribution enters training until its statistics have been
checked. This is a policy change rather than new corpus, but the same rule applies -- the whole point of
the change is a statistic, so measure it first.
"""
import glob
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src/v6")
import train_v8 as V8  # noqa: E402
from fd_fair import native_sizes  # noqa: E402

R = 16
real = sorted({p.replace("\\", "/") for p in
               glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
sz = native_sizes(real)
small = [p for p in real if sz[p] <= R]
mid = [p for p in real if R < sz[p] <= R * 1.5]
rng = random.Random(0)
small = rng.sample(small, min(800, len(small)))
mid = rng.sample(mid, min(800, len(mid)))
print(f"16px bucket under the 1.5x policy: {len([p for p in real if sz[p] <= R])} native, "
      f"{len([p for p in real if R < sz[p] <= R * 1.5])} downscaled", flush=True)


def cov(path, R):
    t = V8.to_tensor(Image.open(path).convert("RGBA"), R)
    a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return float((a[:, :, 3] >= 128).mean())


for m in (0.0, 0.85, 0.8, 0.75, 0.7):
    V8.MARGIN = m
    cs = np.array([cov(p, R) for p in small])
    cm = np.array([cov(p, R) for p in mid])
    # the bucket is the union, weighted by how many sprites each group contributes
    n_s = len([p for p in real if sz[p] <= R])
    n_m = len([p for p in real if R < sz[p] <= R * 1.5])
    mix = (n_s * np.median(cs) + n_m * np.median(cm)) / (n_s + n_m)
    print(f"margin={m or 'off':>5}  native group {np.median(cs):.3f}  downscaled group {np.median(cm):.3f}  "
          f"bucket ~{mix:.3f}   (real sprites 0.262)", flush=True)
