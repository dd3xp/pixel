"""What coverage do the 16 px training targets actually have?

Our samples fill 34% of the canvas where real sprites drawn at 16 px fill 26%. If the targets the model
was trained on fill about 34%, the model is reproducing them faithfully and the fault is upstream; if
the targets fill 26%, the model is doing something of its own. Measured over the three populations that
feed the 16 px bucket under the 1.5x policy.
"""
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src/v6")
from fd_fair import native_sizes  # noqa: E402
from train_cond import to_tensor  # noqa: E402

R, MAX_DOWN = 16, 1.5


def coverage(path, R):
    t = to_tensor(Image.open(path).convert("RGBA"), R)
    a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return float((a[:, :, 3] >= 128).mean())


def native_coverage(path):
    a = np.array(Image.open(path).convert("RGBA"))
    return float((a[:, :, 3] >= 128).mean())


real = sorted({p.replace("\\", "/") for p in
               glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
sz = native_sizes(real)
import random
rng = random.Random(0)

groups = {
    "natively <= 16 px (the medium)": [p for p in real if sz[p] <= R],
    "17-24 px, downscaled into the bucket": [p for p in real if R < sz[p] <= R * MAX_DOWN],
    "25-64 px (refused by the 1.5x policy)": [p for p in real if 25 <= sz[p] <= 64],
}
for name, ps in groups.items():
    ps = rng.sample(ps, min(800, len(ps)))
    cov = np.array([coverage(p, R) for p in ps])
    nat = np.array([native_coverage(p) for p in ps])
    print(f"{name:42s} n={len(ps):4d}  after to_tensor {np.median(cov):.3f}  "
          f"native {np.median(nat):.3f}", flush=True)

# how many sprites are small enough that to_tensor upscales them into the frame?
small = [p for p in real if sz[p] * 2 <= R]
print(f"sprites upscaled by to_tensor (2x or more into a {R} px frame): {len(small)}", flush=True)
