"""Resolution-conditioning ablation: does one ladder model beat per-resolution specialists?

The paper's headline is that resolution-as-condition lets a single model serve
12/16/20/24.  Nothing so far tests whether sharing training across resolutions
HELPS each rung or merely saves us from training four models.  Three runs,
identical data, hyperparameters and step budget (20k), differing only in which
buckets they see:

    abl_ladder : {12,16,20,24}    ~5k steps per rung
    abl_s12    : {12} only        20k steps on that rung
    abl_s16    : {16} only        20k steps on that rung

**The specialists get roughly 4x more gradient steps at their own resolution.**
So if the ladder merely matches them, that is positive transfer across
resolutions; if it beats them, strongly so; and if it loses, the honest reading
is that the ladder buys convenience, not quality.  This asymmetry has to be
stated wherever the numbers are.

Same best-of-8 CLIP reranking as every other comparison in this project, so a
model is never judged on one unlucky sample.

Usage: CUDA_VISIBLE_DEVICES=2 python src/v6/abl_cmp.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

ROOT = Path("/mnt/data/kw/RoundSquisheen/pixel/pixel")
sys.path.insert(0, str(ROOT / "src/v6"))
from sample_v8 import build_model, embed_text, sample, to_rgba  # noqa: E402

N = 8
MODELS = [("ladder", "workdir/abl_ladder/model_latest.pt"),
          ("s12only", "workdir/abl_s12/model_latest.pt"),
          ("s16only", "workdir/abl_s16/model_latest.pt")]


def on_white(im, side=224):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB").resize((side, side), Image.NEAREST)


def main():
    device = "cuda"
    prompts = [l.strip() for l in open(ROOT / "baseline/prompts8.txt", encoding="utf-8")
               if l.strip()]
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    cond = embed_text(prompts, tok, enc, device).repeat_interleave(N, 0)
    uncond = embed_text([""] * len(prompts) * N, tok, enc, device)
    with torch.no_grad():
        t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
        f_t = F.normalize(clip.get_text_features(**t), dim=-1)

    # a specialist only has a claim at its own resolution
    want = {"ladder": [12, 16], "s12only": [12], "s16only": [16]}
    picked, scores = {}, {}
    for name, rel in MODELS:
        ck_path = ROOT / rel
        if not ck_path.exists():
            print(f"missing {rel}, skipping {name}", flush=True)
            continue
        model = build_model(device)
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["unet"] if isinstance(ck, dict) and "unet" in ck else ck)
        model.eval()
        for size in want[name]:
            imgs = [to_rgba(x) for x in
                    sample(model, sched, cond, uncond, size, device, seed=0)]
            out = ROOT / f"runs_out/abl_cmp/{name}_s{size}"
            out.mkdir(parents=True, exist_ok=True)
            best, sc = [], []
            for pi in range(len(prompts)):
                group = imgs[pi * N:(pi + 1) * N]
                with torch.no_grad():
                    px = proc(images=[on_white(g) for g in group], return_tensors="pt").to(device)
                    f_i = F.normalize(clip.get_image_features(**px), dim=-1)
                s = f_i @ f_t[pi]
                k = int(s.argmax())
                group[k].save(out / f"{pi:02d}.png")
                best.append(group[k])
                sc.append(float(s[k]))
            picked[(name, size)] = best
            scores[(name, size)] = sum(sc) / len(sc)
            print(f"{name} s{size}: mean CLIP(best of {N}) = {scores[(name, size)]:.4f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    for size in (12, 16):
        spec = f"s{size}only"
        cols = [c for c in [("ladder", size), (spec, size)] if c in picked]
        if len(cols) < 2:
            continue
        cell, pad, lab = 128, 22, 150
        img = Image.new("RGB", (lab + len(cols) * cell, pad + len(prompts) * cell),
                        (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((4, 4), f"{size}x{size}  " + "  |  ".join(
            f"{n} (CLIP {scores[(n, s)]:.3f})" for n, s in cols), fill=(0, 0, 0))
        for r, p in enumerate(prompts):
            y = pad + r * cell
            d.text((6, y + cell // 2 - 6), p.replace("a pixel art ", "")[:18], fill=(0, 0, 0))
            for c, key in enumerate(cols):
                x = lab + c * cell
                bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
                img.paste(Image.alpha_composite(
                    bg, picked[key][r].resize((cell, cell), Image.NEAREST)).convert("RGB"), (x, y))
                d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(205, 205, 205))
        img.save(ROOT / f"logs/abl_cmp_s{size}.png")
        print(f"-> logs/abl_cmp_s{size}.png", flush=True)


if __name__ == "__main__":
    main()
