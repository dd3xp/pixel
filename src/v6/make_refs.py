"""Generate reference images with SDXL for the image-conditioned path.

These are NOT the output.  They only supply a shape prior: SDXL has seen a
smartphone, our sprite corpus has not.  The reference is encoded by CLIP and
fed to our model, which renders it in pixel-art grammar at 12/16/20/24 px.

Deliberately plain prompts on a white background: at 16px only the silhouette
and palette survive, so a clean isolated object transfers better than a
detailed scene.

Usage: CUDA_VISIBLE_DEVICES=0 python src/v6/make_refs.py prompts/refs.txt data/refs
"""
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

prompts_file, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
lines = [l.strip() for l in open(prompts_file, encoding="utf-8")
         if l.strip() and not l.startswith("#")]

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)

NEG = "photograph, realistic, 3d render, text, watermark, busy background, multiple objects"
for i, p in enumerate(lines):
    full = f"{p}, single centered object, plain white background, simple flat colours, product shot"
    img = pipe(prompt=full, negative_prompt=NEG, num_inference_steps=25,
               guidance_scale=6.0, height=512, width=512,
               generator=torch.Generator("cuda").manual_seed(i)).images[0]
    name = "".join(c if c.isalnum() else "_" for c in p)[:40]
    img.save(out_dir / f"{i:02d}_{name}.png")
    print(f"[{i + 1}/{len(lines)}] {p}", flush=True)
print(f"refs -> {out_dir}", flush=True)
