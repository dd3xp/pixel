"""Extract item-like tiles from Kenney CC0 pixel packs (data/kenney/).
- Roguelike sheets: 16px tiles on a 17px stride (1px gutter).
- Tiny/Platformer packs: individual Tiles/*.png.
Keep only transparent-background objects: alpha fill in [0.08, 0.85] and all
four border rows/cols mostly transparent (drops ground/wall tiles). Output
data/kenney_items/<pack>_<idx>.png (native 16x16 or 18x18 RGBA).

Usage: python src/v6/extract_kenney.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("data/kenney")
OUT = Path("data/kenney_items")
OUT.mkdir(parents=True, exist_ok=True)

SHEETS = {  # pack dir -> (sheet relative path, tile, stride)
    "kenney_roguelike-rpg-pack": ("Spritesheet/roguelikeSheet_transparent.png", 16, 17),
    "kenney_roguelike-caves-dungeons": ("Spritesheet/roguelikeDungeon_transparent.png", 16, 17),
    "kenney_roguelike-indoors": ("Tilesheets/roguelikeIndoor_transparent.png", 16, 17),
}


def keep(tile: Image.Image) -> bool:
    a = np.array(tile.convert("RGBA"))[:, :, 3] > 16
    fill = a.mean()
    if not (0.08 <= fill <= 0.85):
        return False
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return border.mean() < 0.35


total = 0
for pack, (rel, tile, stride) in SHEETS.items():
    sheet = Image.open(SRC / pack / rel).convert("RGBA")
    cols, rows = (sheet.width + 1) // stride, (sheet.height + 1) // stride
    n = 0
    for r in range(rows):
        for c in range(cols):
            t = sheet.crop((c * stride, r * stride, c * stride + tile, r * stride + tile))
            if keep(t):
                t.save(OUT / f"{pack.replace('kenney_', '')}_{r:02d}_{c:02d}.png")
                n += 1
    print(f"{pack}: grid {cols}x{rows} kept {n}", flush=True)
    total += n

for pack in sorted(SRC.glob("kenney_*")):
    if pack.name in SHEETS:
        continue
    n = 0
    for f in sorted(pack.glob("Tiles/*.png")):
        t = Image.open(f).convert("RGBA")
        if keep(t):
            t.save(OUT / f"{pack.name.replace('kenney_', '')}_{f.stem}.png")
            n += 1
    print(f"{pack.name}: kept {n}", flush=True)
    total += n

print(f"TOTAL kept={total}", flush=True)
