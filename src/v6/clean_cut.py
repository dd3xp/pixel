"""Filter junk out of oga_cut: keep sprites, drop glyphs/UI/smoke.
Rules: min side >=10, aspect <=3, alpha fill 0.15-0.97, quantized color count >=4,
not near-monochrome. Output: data/oga_clean/ (copies) + stats.
"""
import glob
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path("data/oga_clean")
OUT.mkdir(parents=True, exist_ok=True)
kept = dropped = 0
for f in glob.glob("data/oga_cut/**/*.png", recursive=True):
    try:
        im = Image.open(f).convert("RGBA")
        w, h = im.size
        if min(w, h) < 10 or max(w, h) / min(w, h) > 3:
            dropped += 1; continue
        a = np.array(im)
        fill = (a[:, :, 3] > 16).mean()
        if not (0.15 <= fill <= 0.97):
            dropped += 1; continue
        vis = a[a[:, :, 3] > 16][:, :3]
        q = (vis // 32)
        ncol = len(np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]))
        if ncol < 4:
            dropped += 1; continue
        if vis.std() < 18:
            dropped += 1; continue
        kept += 1
        shutil.copy(f, OUT / f"{kept:07d}.png")
    except Exception:
        dropped += 1
print(f"kept={kept} dropped={dropped}", flush=True)
