"""Auto-caption cleaned OGA sprites with BLIP for stage-d conditioning.
Sprites are tiny (8-64px); composite onto white, upscale 8x NEAREST first so
BLIP sees crisp pixel shapes. Output: data/oga_captions.csv (path,text).

Usage: CUDA_VISIBLE_DEVICES=0 python src/v6/caption_oga.py
"""
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

SRC = Path("data/oga_clean")
OUT = Path("data/oga_captions.csv")
BS = 64

device = "cuda"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to(device).eval()

done = set()
if OUT.exists():
    with open(OUT, newline="", encoding="utf-8") as f:
        done = {r["path"] for r in csv.DictReader(f)}

paths = [p for p in sorted(SRC.glob("*.png")) if p.name not in done]
print(f"to caption: {len(paths)} (done {len(done)})", flush=True)


def load(p):
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    s = max(im.size)
    return im.resize((im.width * (256 // s or 1), im.height * (256 // s or 1)), Image.NEAREST)


mode = "a" if OUT.exists() else "w"
with open(OUT, mode, newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if mode == "w":
        w.writerow(["path", "text"])
    for i in range(0, len(paths), BS):
        batch = paths[i:i + BS]
        images = [load(p) for p in batch]
        inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        texts = processor.batch_decode(out, skip_special_tokens=True)
        for p, t in zip(batch, texts):
            w.writerow([p.name, t.strip()])
        if (i // BS) % 20 == 0:
            f.flush()
            print(f"[{i + len(batch)}/{len(paths)}] e.g. {texts[0]!r}", flush=True)
print("DONE", flush=True)
