"""Extract small sprite PNGs from OGA-CC0 zips, one zip at a time (disk-safe).
Keeps only images with min(w,h) <= 64 and max(w,h) <= 256 (sprites/sheets of
small cells), discards everything else. Output: data/oga_small/<zip_stem>/...

Usage: python src/v6/extract_oga.py
"""
import io
import zipfile
from pathlib import Path

from PIL import Image

SRC = Path("data/OpenGameArt-CC0")
OUT = Path("data/oga_small")
OUT.mkdir(parents=True, exist_ok=True)

kept = skipped = bad = 0
for zp in sorted(SRC.glob("*.zip")):
    dest = OUT / zp.stem
    if dest.exists():
        print(f"skip {zp.name} (done)", flush=True)
        continue
    dest.mkdir(parents=True)
    n_kept = 0
    try:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".png") or info.file_size > 2_000_000:
                    skipped += 1
                    continue
                try:
                    data = z.read(info)
                    im = Image.open(io.BytesIO(data))
                    w, h = im.size
                    if min(w, h) <= 64 and max(w, h) <= 256:
                        name = f"{n_kept:06d}_{Path(info.filename).name}"
                        (dest / name).write_bytes(data)
                        n_kept += 1
                        kept += 1
                    else:
                        skipped += 1
                except Exception:
                    bad += 1
    except Exception as e:
        print(f"zip error {zp.name}: {e}", flush=True)
    print(f"{zp.name}: kept {n_kept}", flush=True)

print(f"TOTAL kept={kept} skipped={skipped} bad={bad}", flush=True)
