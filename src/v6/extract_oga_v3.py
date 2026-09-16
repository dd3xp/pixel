"""OGA extraction v3: same cutting rules as v2, but over ANY licence bundle, with provenance.

v2 hard-coded the CC0 bundle and threw the source away.  The 09-16 data rebuild needs three more
licence sets (CC-BY-4.0, CC-BY-SA-3.0, OGA-BY-3.0) and a record of where every sprite came from,
both for the paper's licence statement and so a bad source can be dropped without re-extracting.

Two intake paths (v1 kept only the first, v2 only the second):
  * whole PNGs whose longest side is <= 64 px -> kept as-is (these are the *natively* tiny sprites
    the 16 px bucket is starving for);
  * larger PNGs treated as sheets -> alpha connected components of 8..64 px are cropped out.

Usage: python src/v6/extract_oga_v3.py <src_dir_with_2D_Art_zips> <out_dir> <licence_tag>
Writes <out_dir>/<zip stem>/*.png and appends <out_dir>/provenance.csv (path,zip,member,licence,kind).
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SRC, OUT, LIC = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
prov_path = OUT / "provenance.csv"
new_file = not prov_path.exists()
prov = open(prov_path, "a", newline="", encoding="utf-8")
w = csv.writer(prov)
if new_file:
    w.writerow(["path", "zip", "member", "licence", "kind"])

total = 0
for zp in sorted(SRC.glob("2D_Art_*.zip")):
    dest = OUT / zp.stem
    if (dest / ".done").exists():
        print(f"skip {zp.name}", flush=True)
        continue
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".png") or info.file_size > 8_000_000:
                    continue
                try:
                    im = Image.open(io.BytesIO(z.read(info)))
                    if im.mode not in ("RGBA", "LA", "P"):
                        continue
                    rgba = im.convert("RGBA")
                    side = max(rgba.size)
                    if side <= 64:                       # natively tiny sprite: keep whole
                        a = np.array(rgba)[:, :, 3]
                        if side < 8 or not (0.15 <= (a > 16).mean() <= 0.97):
                            continue
                        rgba.save(dest / f"{n:07d}.png")
                        w.writerow([f"{dest.name}/{n:07d}.png", zp.name, info.filename, LIC, "whole"])
                        n += 1
                        continue
                    if side > 2048:
                        continue
                    a = np.array(rgba)[:, :, 3]
                    mask = a > 16
                    if mask.mean() > 0.9 or not mask.any():
                        continue                          # solid image, not a sheet
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
                        if (np.array(crop)[:, :, 3] > 16).mean() < 0.15:
                            continue
                        crop.save(dest / f"{n:07d}.png")
                        w.writerow([f"{dest.name}/{n:07d}.png", zp.name, info.filename, LIC, "cut"])
                        n += 1
                except Exception:
                    pass
    except Exception as e:
        print(f"zip error {zp.name}: {e}", flush=True)
    (dest / ".done").write_text("")
    prov.flush()
    total += n
    print(f"{zp.name}: {n} sprites (total {total})", flush=True)
print(f"TOTAL={total}", flush=True)
