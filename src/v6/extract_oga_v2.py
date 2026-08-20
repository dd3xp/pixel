"""OGA extraction v2: cut sprite SHEETS into individual sprites.
Processes 2D_Art zips only; PNGs 64-2048px with alpha are labeled via
scipy.ndimage on the alpha channel; connected components sized 8-64px are
cropped (2px pad) and saved. Complements v1 (which kept only small whole PNGs).

Usage: python src/v6/extract_oga_v2.py
"""
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SRC = Path("data/OpenGameArt-CC0")
OUT = Path("data/oga_cut")
OUT.mkdir(parents=True, exist_ok=True)

total = 0
for zp in sorted(SRC.glob("2D_Art_*.zip")):
    dest = OUT / zp.stem
    if dest.exists():
        print(f"skip {zp.name}", flush=True)
        continue
    dest.mkdir(parents=True)
    n = 0
    try:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".png") or info.file_size > 8_000_000:
                    continue
                try:
                    im = Image.open(io.BytesIO(z.read(info)))
                    w, h = im.size
                    if not (64 < max(w, h) <= 2048) or im.mode not in ("RGBA", "LA", "P"):
                        continue
                    rgba = im.convert("RGBA")
                    a = np.array(rgba)[:, :, 3]
                    mask = a > 16
                    if mask.mean() > 0.9 or not mask.any():
                        continue  # solid image, not a sprite sheet
                    labels, k = ndimage.label(mask)
                    if k < 2 or k > 4000:
                        continue
                    for sl in ndimage.find_objects(labels):
                        ch, cw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                        if not (8 <= ch <= 64 and 8 <= cw <= 64):
                            continue
                        y0 = max(0, sl[0].start - 2); y1 = min(rgba.height, sl[0].stop + 2)
                        x0 = max(0, sl[1].start - 2); x1 = min(rgba.width, sl[1].stop + 2)
                        crop = rgba.crop((x0, y0, x1, y1))
                        ca = np.array(crop)[:, :, 3]
                        if (ca > 16).mean() < 0.15:
                            continue  # mostly empty crop
                        crop.save(dest / f"{n:07d}.png")
                        n += 1
                except Exception:
                    pass
    except Exception as e:
        print(f"zip error {zp.name}: {e}", flush=True)
    total += n
    print(f"{zp.name}: cut {n} (total {total})", flush=True)

print(f"TOTAL cut={total}", flush=True)
