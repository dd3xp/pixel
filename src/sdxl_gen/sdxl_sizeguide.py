"""Generality test of cross-resolution self-guidance on a public model: SDXL's size micro-conditioning.

SDXL receives `original_size` as a conditioning signal (Podell et al. 2023); the paper documents that a small
original_size makes the model reproduce the look of low-resolution training images ("simpler patterns, blurring").
That is the SDXL analogue of our lower bucket label. On the same latent x_t and the same caption we query:

  e_c      = eps(x_t, c,   orig=1024)          strong, conditional
  e_u      = eps(x_t, "",  orig=1024)          CFG's unconditional reference
  e_w      = eps(x_t, c,   orig=LOW)           ours: same caption, lower "resolution" label
  e_uw     = eps(x_t, "",  orig=LOW)           the diffusers `negative_original_size` practice (drops the caption too)

Modes (--mode):
  cfg        e = e_u  + w (e_c - e_u)                               2 NFE, the baseline, sweep w
  negos      e = e_uw + w (e_c - e_uw)                              2 NFE, existing practice (our `bucketu` analogue)
  label      e = e_w  + w (e_c - e_w)                               2 NFE, ours, no text term at all
  cfglabel   e = e_u  + w (e_c - e_u) + lam (e_c - e_w)             3 NFE, ours added on top of CFG
The pixel-art model needed almost no CFG (best w = 1.5), SDXL needs w ~ 5, so `label` alone is expected to lose
alignment; `cfglabel` is the fair form, and `negos` is the direct competitor (same idea minus the caption).

Output: <out>/<idx:05d>.png at --save_px (default 512), resumable, shardable with --start/--end.
Usage: python src/sdxl_gen/sdxl_sizeguide.py --prompts data/coco30k/captions.txt --mode cfg --w 5 --out runs_out/sdxl/cfg5
"""
import argparse
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def time_ids(orig, target, n, dev):
    return torch.tensor([[orig, orig, 0, 0, target, target]], device=dev, dtype=torch.float16).repeat(n, 1)


@torch.no_grad()
def generate(pipe, sched, prompts, a, seed, dev):
    B = len(prompts)
    pe, npe, ppe, nppe = pipe.encode_prompt(prompt=prompts, device=dev, num_images_per_prompt=1,
                                            do_classifier_free_guidance=True)
    hi = time_ids(a.res, a.res, B, dev)
    lo = time_ids(a.low, a.res, B, dev)
    # branch table: name -> (text, pooled, time_ids)
    table = {"c": (pe, ppe, hi), "u": (npe, nppe, hi), "w": (pe, ppe, lo), "uw": (npe, nppe, lo)}
    need = {"cfg": ["c", "u"], "negos": ["c", "uw"], "label": ["c", "w"], "cfglabel": ["c", "u", "w"]}[a.mode]
    enc = torch.cat([table[k][0] for k in need])
    pool = torch.cat([table[k][1] for k in need])
    tids = torch.cat([table[k][2] for k in need])

    sched.set_timesteps(a.steps, device=dev)
    # one generator per prompt index: every mode, batch size and resume sees the same noise for the same prompt
    lat = torch.cat([torch.randn((1, 4, a.res // 8, a.res // 8), device=dev, dtype=torch.float32,
                                 generator=torch.Generator(device=dev).manual_seed(s)) for s in seed])
    lat = lat * sched.init_noise_sigma
    for t in sched.timesteps:
        xin = sched.scale_model_input(lat, t).to(torch.float16)
        out = pipe.unet(xin.repeat(len(need), 1, 1, 1), t, encoder_hidden_states=enc,
                        added_cond_kwargs={"text_embeds": pool, "time_ids": tids}).sample.float()
        e = dict(zip(need, out.chunk(len(need))))
        if a.mode == "cfg":
            eps = e["u"] + a.w * (e["c"] - e["u"])
        elif a.mode == "negos":
            eps = e["uw"] + a.w * (e["c"] - e["uw"])
        elif a.mode == "label":
            eps = e["w"] + a.w * (e["c"] - e["w"])
        else:
            eps = e["u"] + a.w * (e["c"] - e["u"]) + a.lam * (e["c"] - e["w"])
        lat = sched.step(eps, t, lat).prev_sample
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample      # VAE kept in fp32 (fp16 overflows)
    return ((img + 1) / 2).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="cfg", choices=["cfg", "negos", "label", "cfglabel"])
    ap.add_argument("--w", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the size term in cfglabel")
    ap.add_argument("--low", type=int, default=512, help="original_size of the weak reference")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_px", type=int, default=512)
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [l.rstrip("\n") for l in open(a.prompts, encoding="utf-8")]
    todo = [i for i in range(a.start, min(a.end, len(prompts))) if not (out / f"{i:05d}.png").exists()]
    print(f"{a.mode} w={a.w} lam={a.lam} low={a.low}: {len(todo)} to generate -> {out}", flush=True)
    if not todo:
        print("SDXL_SIZEGUIDE_DONE", out, flush=True); return

    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to(dev)
    pipe.vae.to(torch.float32)
    pipe.set_progress_bar_config(disable=True)
    sched = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    from PIL import Image
    for k in range(0, len(todo), a.bs):
        idx = todo[k:k + a.bs]
        imgs = generate(pipe, sched, [prompts[i] for i in idx], a, [a.seed * 100000 + i for i in idx], dev)
        for i, im in zip(idx, imgs):
            arr = (im.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            Image.fromarray(arr).resize((a.save_px, a.save_px), Image.BICUBIC).save(out / f"{i:05d}.png")
        if (k // a.bs) % 25 == 0:
            print(f"[{k + len(idx)}/{len(todo)}]", flush=True)
    print("SDXL_SIZEGUIDE_DONE", out, flush=True)


if __name__ == "__main__":
    main()
