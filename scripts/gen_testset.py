"""Generate a diverse test set with SDXL (server-side, cached weights) + ellipse masks.
Usage: python scripts/gen_testset.py
"""
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFilter

PROMPTS = [
    ("ts1_apple", "a shiny red apple with a green leaf, plain light gray background, centered, product photo"),
    ("ts2_cat", "a cute ginger cat sitting upright, plain pale blue background, centered"),
    ("ts3_mushroom", "a red mushroom with white spots, plain beige background, centered"),
    ("ts4_chest", "a wooden treasure chest with gold trim, plain dark background, centered"),
    ("ts5_cactus", "a green cactus in a terracotta pot, plain warm background, centered"),
    ("ts6_fish", "a tropical orange and white clownfish, plain deep blue background, centered"),
]


def main() -> None:
    out = Path("assets/test_images")
    out.mkdir(parents=True, exist_ok=True)
    mask_dir = Path("assets/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    ).to("cuda")
    g = torch.Generator("cuda").manual_seed(42)
    for name, prompt in PROMPTS:
        img = pipe(prompt, num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
        img.save(out / f"{name}.png")
        m = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(m).ellipse((160, 130, 870, 900), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(20))
        m.save(mask_dir / f"{name}_mask.png")
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
