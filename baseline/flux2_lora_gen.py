"""External baseline: FLUX.2-klein-4B + the Limbicnation/pixel-art-lora pixel-sprite LoRA (2026, Apache-2.0).

The newest open pixel-sprite model at the time of writing, and the direct successor question to the SDXL +
Pixel Art XL baseline: "does a 2026 4B rectified-flow model with a purpose-trained sprite LoRA close the gap?".

Same protocol as the other "generate big, then shrink" baselines (api_gen.py / sdxl_lora_gen.py): the first N
recaptioned held-out prompts, the same SUFFIX, one image per prompt with a per-index seed, saved as %05d.png.
The renders then go through api_to_sprites.py (cutout + the project's own to_tensor), so the FD is computed on
exactly the same footing as gpt-image-2 / nano-banana-2 / SDXL+LoRA and as our own samples.

Model card defaults (https://hf-mirror.com/black-forest-labs/FLUX.2-klein-4B): Flux2KleinPipeline, bf16,
4 steps, guidance_scale 1.0.  LoRA card (https://hf-mirror.com/Limbicnation/pixel-art-lora): weight file
pytorch_lora_weights.safetensors, 512x512, 4 steps, CFG 1.0, LoRA strength 0.85-1.4, and the trigger phrase

    "pixel art sprite, [character description], game asset, transparent background"

which is what TRIGGER_HEAD/TRIGGER_TAIL wrap the prompt in below; the project SUFFIX is appended after it so the
prompt carries the same "single object / centred / no text" constraints as every other baseline.

LoRA strength: the safetensors ships no `.alpha` tensors, but the repo's config.json (rank 64, alpha 128,
use_rslora) is picked up as if it were a peft adapter config, so peft scales the LoRA by alpha/sqrt(rank) = 16.
At that strength the model collapses to black noise (verified: mean pixel 2/255 at every prompt tried).  A plain
LoRA loader — ComfyUI, and the strength 0.85-1.4 the card quotes — has no alpha metadata either and therefore
applies the standard alpha=rank scaling, i.e. a plain multiplier equal to the strength.  --lora_strength is that
plain multiplier and is divided by whatever factor peft inferred, so --lora_strength 1.0 reproduces the card's
setting and the sprites in its sample gallery.

The card claims "512x512 RGBA output with transparent backgrounds", but the FLUX.2 VAE has out_channels=3 and the
pipeline returns RGB: the model *paints* a grey/white checkerboard where it means transparency.  Images are saved
exactly as returned, so a later pass can decide how to treat that background.

Usage: python baseline/flux2_lora_gen.py --out runs_out/ext200/runs/ext_flux2/big --n 200
"""
import argparse
import math
import sys
from pathlib import Path

import torch

if torch.__version__ < "2.5":
    # diffusers 0.37 guards its torch.library.custom_op registrations with `torch.__version__ >= "2.4.0"`, but
    # attention_dispatch.py has `from __future__ import annotations`, and torch 2.4's infer_schema() cannot read
    # string annotations -> importing any pipeline dies with "Parameter q has unsupported type torch.Tensor".
    # Select the no-op fallback diffusers itself uses for unsupported torch versions; it only skips the custom-op
    # registration of the flash-attn 2/3 wrappers, which this script never uses (default backend is native SDPA).
    def _no_op(name, fn=None, /, *, mutates_args=None, device_types=None, schema=None, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        return wrap if fn is None else fn

    torch.library.custom_op = _no_op
    torch.library.register_fake = _no_op

    # FLUX.2 uses grouped-query attention and diffusers calls SDPA with enable_gqa=True, which torch only grew in
    # 2.5.  The flag is pure sugar for repeating the kv heads up to the number of q heads, so do exactly that.
    _sdpa = torch.nn.functional.scaled_dot_product_attention

    def _sdpa_gqa(query, key, value, *args, enable_gqa=False, **kwargs):
        if enable_gqa:
            n_rep = query.size(-3) // key.size(-3)
            if n_rep > 1:
                key = key.repeat_interleave(n_rep, dim=-3)
                value = value.repeat_interleave(n_rep, dim=-3)
        return _sdpa(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = _sdpa_gqa

sys.path.insert(0, "baseline")
from api_gen import SUFFIX  # noqa: E402

TRIGGER_HEAD = "pixel art sprite, "
TRIGGER_TAIL = ", game asset, transparent background"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="runs_out/heldout3000_prompts_recap.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=512)      # the LoRA is trained at 512
    ap.add_argument("--steps", type=int, default=4)       # klein is a 4-step distilled model
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--lora_strength", type=float, default=1.0, help="plain LoRA multiplier, as on the card")
    ap.add_argument("--offload", action="store_true", help="enable_model_cpu_offload() if GPU memory is tight")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()][:a.n]
    todo = [i for i in range(len(prompts)) if not (out / f"{i:05d}.png").exists()]
    print(f"{len(todo)} to generate -> {out}", flush=True)
    if not todo:
        return

    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights("Limbicnation/pixel-art-lora", weight_name="pytorch_lora_weights.safetensors")
    name = next(iter(pipe.transformer.peft_config))
    cfg = pipe.transformer.peft_config[name]
    inferred = cfg.lora_alpha / (math.sqrt(cfg.r) if getattr(cfg, "use_rslora", False) else cfg.r)
    pipe.set_adapters([name], [a.lora_strength / inferred])
    print(f"LoRA r={cfg.r} alpha={cfg.lora_alpha} rslora={getattr(cfg, 'use_rslora', False)} -> peft scale "
          f"{inferred:g}; applying {a.lora_strength:g}/{inferred:g} = {a.lora_strength / inferred:g}", flush=True)
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for k, i in enumerate(todo):
        g = torch.Generator("cuda").manual_seed(i)
        text = TRIGGER_HEAD + prompts[i] + TRIGGER_TAIL + SUFFIX
        img = pipe(prompt=text, height=a.size, width=a.size, num_inference_steps=a.steps,
                   guidance_scale=a.cfg, generator=g).images[0]
        img.save(out / f"{i:05d}.png")
        if k % 20 == 0:
            print(f"[{k + 1}/{len(todo)}] mode={img.mode}", flush=True)
    print("FLUX2_LORA_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
