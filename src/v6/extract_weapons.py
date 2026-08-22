"""Extract item sprites from downloaded CC0 weapon/item packs (data/oga_weapons/).
Unzips every zip, then for every PNG: if max side <= 64 keep whole; else cut
alpha-connected components 8-64px (2px pad). Hollow filter: alpha fill
0.08-0.97 and border mostly transparent. Output data/weapon_items/.

Usage: python src/v6/extract_weapons.py
"""
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SRC = Path("data/oga_weapons")
OUT = Path("data/weapon_items")
OUT.mkdir(parents=True, exist_ok=True)

for zp in SRC.glob("*.zip"):
    d = SRC / ("unz_" + zp.stem)
    if not d.exists():
        d.mkdir()
        try:
            zipfile.ZipFile(zp).extractall(d)
        except Exception as e:
            print("bad zip", zp.name, e)


def hollow(rgba: Image.Image) -> bool:
    a = np.array(rgba)[:, :, 3] > 16
    fill = a.mean()
    if not (0.08 <= fill <= 0.97):
        return False
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).mean()
    return border < 0.35


n = 0
for png in SRC.rglob("*.png"):
    if "preview" in png.name.lower():
        continue
    try:
        im = Image.open(png).convert("RGBA")
    except Exception:
        continue
    w, h = im.size
    stem = png.stem.replace(" ", "_")[:40]
    if max(w, h) <= 64:
        if hollow(im):
            im.save(OUT / f"{stem}_{n:05d}.png")
            n += 1
        continue
    a = np.array(im)[:, :, 3]
    mask = a > 16
    if mask.mean() > 0.9 or not mask.any():
        continue
    labels, k = ndimage.label(mask)
    if k < 1 or k > 4000:
        continue
    for sl in ndimage.find_objects(labels):
        ch, cw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (8 <= ch <= 64 and 8 <= cw <= 64):
            continue
        y0 = max(0, sl[0].start - 2); y1 = min(h, sl[0].stop + 2)
        x0 = max(0, sl[1].start - 2); x1 = min(w, sl[1].stop + 2)
        crop = im.crop((x0, y0, x1, y1))
        if hollow(crop):
            crop.save(OUT / f"{stem}_{n:05d}.png")
            n += 1
print(f"TOTAL kept={n}", flush=True)
