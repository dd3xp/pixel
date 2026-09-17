"""Apply the same k-colour median-cut post-process to the external baselines' sprites.

The palette projection is part of our sampler, so the comparison is only fair if the baselines are allowed
the same treatment on their finished sprites (09-17).  Usage: python src/v6/quant_dirs.py <R> <k>
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

R, K = sys.argv[1], int(sys.argv[2])
SRC = {"gpt": f"runs_out/ext200/runs/ext_recap_gpt/s{R}",
       "flux2": f"runs_out/ext200/runs/ext_flux2/s{R}",
       "lora": f"runs_out/ext200/runs/ext_lora/s{R}",
       "real": f"runs_out/ext200/ext200_ours/real/s{R}"}
for tag, src in SRC.items():
    out = f"runs_out/ext200/q/{tag}{R}_q{K}/s{R}"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(src + "/*.png"))
    if not files or len(glob.glob(out + "/*.png")) >= len(files):
        continue
    for p in files:
        a = np.array(Image.open(p).convert("RGBA"))
        al = a[:, :, 3]
        rgb = Image.fromarray(a[:, :, :3], "RGB").quantize(colors=K, method=Image.MEDIANCUT,
                                                           dither=Image.Dither.NONE).convert("RGB")
        b = np.dstack([np.array(rgb), al])
        b[al == 0] = 0
        Image.fromarray(b, "RGBA").save(os.path.join(out, os.path.basename(p)))
    print("wrote", out, flush=True)
