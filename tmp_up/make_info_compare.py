"""Same informative caption down each column: ours / gpt-image-2 downscaled / SDXL downscaled.
No real-sprite row exists here because these prompts were written by hand, not derived from artwork."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R, CELL, PAD, LABEL_W, NCOL = 16, 104, 5, 250, 16
prompts = [l.strip() for l in open("runs_out/prompts_informative.txt", encoding="utf-8") if l.strip()]
def ours(c): return f"runs_out/info_ours/s{R}/{c:02d}_0.png"
def gpt(c):  return f"runs_out/info_gpt/s{R}/{c:05d}.png"
def sdxl(c):
    p = f"runs_out/info_sdxl/s{R}/{c:02d}_0.png"
    return p if Path(p).exists() else f"runs_out/info_sdxl/s{R}/{c:05d}.png"
ROWS = [("OURS: composed guidance", ours), ("gpt-image-2 -> downscaled", gpt), ("SDXL-base -> downscaled", sdxl)]

def checker(w, h, k=8):
    a = np.indices((h, w)).sum(0) // k % 2
    return Image.fromarray((235 + 12 * a).astype(np.uint8)).convert("RGBA")

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15); small = ImageFont.truetype("DejaVuSans.ttf", 9)
except OSError:
    font = small = ImageFont.load_default()

for page in range(2):
    cols = [c for c in range(page * NCOL, min((page + 1) * NCOL, len(prompts)))
            if all(Path(f(c)).exists() for _, f in ROWS)]
    if not cols: continue
    W = LABEL_W + len(cols) * (CELL + PAD) + PAD
    H = len(ROWS) * (CELL + PAD) + PAD + 60
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255)); d = ImageDraw.Draw(canvas)
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
        words = prompts[c].split()
        for li, ln in enumerate([" ".join(words[i:i+3]) for i in range(0, min(len(words), 9), 3)]):
            d.text((LABEL_W + PAD + k * (CELL + PAD), y + 3 + li * 11), ln, fill=(60, 60, 60, 255), font=small)
    out = f"runs_out/fig_info_compare_{page}.png"
    canvas.convert("RGB").save(out); print("wrote", out, canvas.size)
