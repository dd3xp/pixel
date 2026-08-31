"""v7c vs the two distillation fine-tunes, on the same objects and seed.

CLIP says both v11 variants are worse everywhere -- including on the 49 objects
whose pseudo-labels they trained on -- but CLIP already disagreed with the eye
once in this project (the ablation), so the claim is not settled until the
sprites are looked at.

Rows are chosen to answer the two questions at once: the in-domain controls show
whether the craft was damaged, and objects from the weak categories show whether
any vocabulary was gained.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
MODELS = [("v7c", "runs_out/coverage16/best"),
          ("r15", "runs_out/coverage16_r15/best"),
          ("v12", "runs_out/coverage16_v12/best")]
# 5 in-domain controls, then objects from the categories coverage found weak
WANT = ["golden_sword", "iron_pickaxe", "health_potion", "gold_coin", "red_apple",
        "smartphone", "laptop_computer", "coffee_machine", "camera", "scissors",
        "acoustic_guitar", "soccer_ball", "sneaker", "microscope", "telescope"]

cell, pad, lab = 108, 22, 160
img = Image.new("RGB", (lab + len(MODELS) * cell, pad + len(WANT) * cell), (245, 245, 245))
d = ImageDraw.Draw(img)
d.text((4, 4), "  |  ".join(n for n, _ in MODELS) + "   (rows 1-5 = in-domain controls)",
       fill=(0, 0, 0))

for r, obj in enumerate(WANT):
    y = pad + r * cell
    d.text((6, y + cell // 2 - 6), obj[:20], fill=(0, 0, 0))
    for c, (name, rel) in enumerate(MODELS):
        hits = sorted((ROOT / rel).glob(f"*_{obj}.png"))
        x = lab + c * cell
        if not hits:
            d.text((x + 30, y + cell // 2), "n/a", fill=(160, 0, 0))
            continue
        bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
        im = Image.open(hits[0]).convert("RGBA").resize((cell, cell), Image.NEAREST)
        img.paste(Image.alpha_composite(bg, im).convert("RGB"), (x, y))
        d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(205, 205, 205))

img.save(ROOT / "logs/v11_cmp.png")
print("-> logs/v11_cmp.png")
