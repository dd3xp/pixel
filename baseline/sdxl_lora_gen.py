"""External baseline: SDXL base 1.0 + the Pixel Art XL LoRA (nerijs/pixel-art-xl, the most-used open pixel-art model).

Same protocol as the other "generate big, then shrink" baselines (api_gen.py): the first N recaptioned held-out
prompts with the same suffix (single object, plain white background), 1024 px, one image per prompt with a per-index
seed. The renders then go through api_to_sprites.py (cutout + the project's own to_tensor) and pixeloe_sprites.py,
exactly like gpt-image-2 / nano-banana-2. The LoRA card recommends no refiner and no trigger word.
Usage: python baseline/sdxl_lora_gen.py --out runs_out/ext200/runs/ext_lora/big --n 200
"""
import argparse
import sys
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

sys.path.insert(0, "baseline")
from api_gen import SUFFIX  # noqa: E402


@torch.no_grad()      # the manual VAE decode below otherwise builds a graph and .numpy() fails (09-15)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="runs_out/heldout3000_prompts_recap.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()][:a.n]
    todo = [i for i in range(len(prompts)) if not (out / f"{i:05d}.png").exists()]
    print(f"{len(todo)} to generate -> {out}", flush=True)
    if not todo:
        return
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                     torch_dtype=torch.float16, use_safetensors=True)
    pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    pipe.fuse_lora(lora_scale=a.lora_scale)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.vae.to(torch.float32)          # SDXL VAE overflows in fp16
    pipe.set_progress_bar_config(disable=True)
    for k, i in enumerate(todo):
        g = torch.Generator("cuda").manual_seed(i)
        lat = pipe(prompts[i] + SUFFIX, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g,
                   output_type="latent").images
        img = pipe.vae.decode(lat.float() / pipe.vae.config.scaling_factor).sample
        img = ((img[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        Image.fromarray((img * 255).round().astype("uint8")).save(out / f"{i:05d}.png")
        if k % 20 == 0:
            print(f"[{k + 1}/{len(todo)}]", flush=True)
    print("SDXL_LORA_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
