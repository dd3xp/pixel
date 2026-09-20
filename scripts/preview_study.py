import json
from pathlib import Path
from PIL import Image, ImageDraw
d = Path("runs_out/human_study")
t = json.load(open(d / "trials.json", encoding="utf-8"))
idxs = sorted({r["index"] for r in t["trials"]})[:10]
sysl = ["real", "ours", "flux", "gpt", "sdxl"]
cell, pad, lab = 128, 4, 46
W = lab + len(idxs) * (cell + pad)
H = len(sysl) * (cell + pad)
sheet = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
for r, k in enumerate(sysl):
    dr.text((4, r * (cell + pad) + cell // 2), k, fill=(0, 0, 0))
    for c, i in enumerate(idxs):
        sheet.paste(Image.open(d / "img" / f"{k}_{i:05d}.png"), (lab + c * (cell + pad), r * (cell + pad)))
sheet.save(d / "preview.png")
print("ok", W, H)
