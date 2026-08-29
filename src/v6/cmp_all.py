"""Apples-to-apples: ours vs SDXL-then-downscale, both best-of-8 by CLIP.

The downscale baseline picks its best of 8 candidates, so ours must get the same
treatment or the comparison is rigged in our favour by nothing more than sample
count.  Same reranker (CLIP ViT-B/32, prompt vs sprite composited on white),
same N, same prompts, same seeds policy.

Writes:
  runs_out/v7c_best8/s{n}/{i:02d}.png   our best-of-8 per prompt
  logs/cmp_all_s{n}.png                 rows = prompts, cols = ours | SD-piXL | downscale
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
sys.path.insert(0, str(ROOT / "src/v6"))
from sample_v8 import build_model, embed_text, sample, to_rgba  # noqa: E402

CKPT = ROOT / "workdir/v7c_bow/model_latest.pt"
SIZES = [int(a) for a in sys.argv[1:]] or [12, 16]
N = 8
DS = "mean_raw"          # strongest downscale variant; db32 quantisation hurt


def on_white(rgba_img, side=224):
    bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba_img).convert("RGB").resize((side, side), Image.NEAREST)


def main():
    device = "cuda"
    prompts = [l.strip() for l in open(ROOT / "baseline/prompts8.txt", encoding="utf-8")
               if l.strip()]
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = build_model(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["unet"] if isinstance(ck, dict) and "unet" in ck else ck)
    model.eval()
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    cond = embed_text(prompts, tok, enc, device).repeat_interleave(N, 0)
    uncond = embed_text([""] * len(prompts) * N, tok, enc, device)
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        f_t = F.normalize(clip.get_text_features(**t), dim=-1)

    for n in SIZES:
        imgs = [to_rgba(x) for x in sample(model, sched, cond, uncond, n, device, seed=0)]
        outdir = ROOT / f"runs_out/v7c_best8/s{n}"
        outdir.mkdir(parents=True, exist_ok=True)
        best = []
        for pi in range(len(prompts)):
            group = imgs[pi * N:(pi + 1) * N]
            with torch.no_grad():
                px = proc(images=[on_white(g) for g in group],
                          return_tensors="pt").to(device)
                f_i = F.normalize(clip.get_image_features(**px), dim=-1)
            k = int((f_i @ f_t[pi]).argmax())
            group[k].save(outdir / f"{pi:02d}.png")
            best.append(group[k])
            print(f"s{n} p{pi}: best sample {k} ({prompts[pi]})", flush=True)

        cell, pad, lab = 128, 22, 150
        cols = 3
        out = Image.new("RGB", (lab + cols * cell, pad + len(prompts) * cell), (245, 245, 245))
        d = ImageDraw.Draw(out)
        d.text((4, 4), f"{n}x{n}   ours (best of {N})  |  SD-piXL 10k  |  SDXL+downscale ({DS}, best of {N})",
               fill=(0, 0, 0))
        for r, p in enumerate(prompts):
            y = pad + r * cell
            d.text((6, y + cell // 2 - 6), p.replace("a pixel art ", "")[:18], fill=(0, 0, 0))
            srcs = [outdir / f"{r:02d}.png",
                    ROOT / f"baseline/results/10k_s{n}_p{r + 1}.png",
                    ROOT / f"runs_out/dsbaseline3/s{n}/{DS}/{r:02d}.png"]
            for c, s in enumerate(srcs):
                x = lab + c * cell
                if not Path(s).exists():
                    d.text((x + 40, y + cell // 2), "n/a", fill=(160, 0, 0))
                    continue
                im = Image.open(s).convert("RGBA").resize((cell, cell), Image.NEAREST)
                bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
                out.paste(Image.alpha_composite(bg, im).convert("RGB"), (x, y))
                d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(205, 205, 205))
        out.save(ROOT / f"logs/cmp_all_s{n}.png")
        print(f"-> logs/cmp_all_s{n}.png", flush=True)


if __name__ == "__main__":
    main()
