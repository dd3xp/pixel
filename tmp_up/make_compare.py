"""Prompt-aligned comparison strip: the same held-out caption down each column.
Rows: real held-out sprite / our composed guidance / gpt-image-2 downscaled / SDXL downscaled."""
import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R, NCOL, CELL, PAD, LABEL_W = 16, 16, 96, 4, 300
OUT = "runs_out/fig_compare_external.png"
prompts = [l.strip() for l in open("runs_out/heldout3000_prompts.txt", encoding="utf-8") if l.strip()]

def real(c):   return f"runs_out/heldout3000_totensor_s{R}/{c:05d}.png"
def ours(c):   return f"runs_out/v7h_gbk12s10k_w1p5_matched_eval/s{R}/{c:02d}_0.png"
def gpt(c):    return f"runs_out/api_gpt_image_2/s{R}/{c:05d}.png"
def sdxl(c):
    p = f"runs_out/sdxl_dn_eval/s{R}/{c:02d}_0.png"
    return p if Path(p).exists() else f"runs_out/sdxl_dn_eval/s{R}/{c:05d}.png"

ROWS = [("real held-out sprite", real), ("OURS: composed guidance", ours),
        ("gpt-image-2 -> downscaled", gpt), ("SDXL-base -> downscaled", sdxl)]

# keep only columns where every row exists
cols = [c for c in range(400) if all(Path(f(c)).exists() for _, f in ROWS)][:NCOL]
print("columns:", cols)

def checker(w, h, k=8):
    a = np.indices((h, w)).sum(0) // k % 2
    return Image.fromarray((235 + 12 * a).astype(np.uint8)).convert("RGBA")

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    small = ImageFont.truetype("DejaVuSans.ttf", 10)
except OSError:
    font = small = ImageFont.load_default()

W = LABEL_W + len(cols) * (CELL + PAD) + PAD
H = len(ROWS) * (CELL + PAD) + PAD + 46
canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
d = ImageDraw.Draw(canvas)
for r, (label, fn) in enumerate(ROWS):
    y = PAD + r * (CELL + PAD)
    d.text((8, y + CELL // 2 - 8), label, fill=(0, 0, 0, 255), font=font)
    for k, c in enumerate(cols):
        x = LABEL_W + PAD + k * (CELL + PAD)
        bg = checker(CELL, CELL)
        bg.alpha_composite(Image.open(fn(c)).convert("RGBA").resize((CELL, CELL), Image.NEAREST))
        canvas.paste(bg, (x, y))
y = PAD + len(ROWS) * (CELL + PAD)
for k, c in enumerate(cols):
    t = prompts[c][:26]
    d.text((LABEL_W + PAD + k * (CELL + PAD), y + 4), t, fill=(60, 60, 60, 255), font=small)
canvas.convert("RGB").save(OUT)
print("wrote", OUT, canvas.size)
