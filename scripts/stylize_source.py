"""Stage-0 stylization: repaint the source as a high-res PIXEL-ART-STYLE image
(dark outlines, flat shading, simplified shapes) before the pyramid ever runs.
Structure is held by ControlNet canny; style comes from the pixel-art LoRA.

Usage: python scripts/stylize_source.py <src> "<prompt>" <out> [strength]
"""
import sys

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
from PIL import Image


def main() -> None:
    src_path, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.65

    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0-mid", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    pipe.fuse_lora()

    img = Image.open(src_path).convert("RGB").resize((1024, 1024))
    arr = np.array(img)
    edges = cv2.Canny(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), 100, 200)
    control = Image.fromarray(np.stack([edges] * 3, -1))

    out = pipe(
        prompt=prompt,
        image=img,
        control_image=control,
        strength=strength,
        controlnet_conditioning_scale=0.5,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]
    # flatten fine dither texture (it aliases into streaky noise when downsampled);
    # median filter keeps outlines and flat regions, kills pixel-level texture
    flat = cv2.medianBlur(np.array(out), 7)
    Image.fromarray(flat).save(out_path)
    print(f"stylized -> {out_path}")


if __name__ == "__main__":
    main()
