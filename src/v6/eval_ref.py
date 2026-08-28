"""Objective metric for the image-conditioned path: does the reference's shape
actually reach the sprite?

Human scoring of 12 references x 4 samples turned out noisier than the effect
we were trying to measure (the v8 out-of-domain score wandered 4.4-5.3 across
checkpoints with no reliable trend).  This replaces the eyeball with:

  ref_sim  = cos( CLIP_img(reference) , CLIP_img(generated) )
  txt_sim  = cos( CLIP_txt(prompt)    , CLIP_img(generated) )

ref_sim is the transfer measure; txt_sim guards against a degenerate model
that copies the reference but loses the caption's meaning.  Both are averaged
over refs x samples, with a fixed seed so checkpoints are comparable.

Usage:
  python src/v6/eval_ref.py --ckpts workdir/v8_imgcond/model_step0200*.pt ... \
      --refs data/refs --prompts prompts/refs.txt --size 16 --n 8 --out workdir/refeval.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from sample_v8 import (BUCKETS, build_model, embed_text, load_ref, sample,  # noqa: E402
                       to_rgba, CLIP_MEAN, CLIP_STD)
from diffusers import DDPMScheduler  # noqa: E402
from transformers import (CLIPModel, CLIPTextModel, CLIPTokenizer,  # noqa: E402
                          CLIPVisionModel)


def clip_view(pil):
    """RGBA sprite -> CLIP input tensor (white background, 224px)."""
    bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, pil).convert("RGB").resize((224, 224), Image.NEAREST)
    import numpy as np
    t = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
    return (t - CLIP_MEAN) / CLIP_STD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--refs", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="workdir/refeval.csv")
    p.add_argument("--text_only", action="store_true",
                   help="checkpoints are plain v7 state dicts; measures the no-image control")
    args = p.parse_args()
    device = "cuda"

    ref_paths = sorted([q for q in Path(args.refs).iterdir()
                        if q.suffix.lower() in (".png", ".jpg", ".jpeg")])
    texts = [l.strip() for l in open(args.prompts, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    assert len(texts) == len(ref_paths), f"{len(texts)} prompts vs {len(ref_paths)} refs"

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    ref_px = torch.stack([load_ref(q) for q in ref_paths]).to(device)
    with torch.no_grad():
        ref_feat = F.normalize(clip.get_image_features(pixel_values=ref_px), dim=-1)
        tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        txt_feat = F.normalize(clip.get_text_features(**tok), dim=-1)

    rows = []
    for ckpt in args.ckpts:
        model = build_model(device)
        img_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 512)).to(device)
        ck = torch.load(ckpt, map_location=device)
        if args.text_only:
            model.load_state_dict(ck if not isinstance(ck, dict) or "unet" not in ck else ck["unet"])
        else:
            model.load_state_dict(ck["unet"])
            img_proj.load_state_dict(ck["img_proj"])
        model.eval(); img_proj.eval()

        cond = embed_text(texts, tokenizer, enc, device).repeat_interleave(args.n, 0)
        uncond = embed_text([""] * len(texts) * args.n, tokenizer, enc, device)
        if args.text_only:
            imgs = sample(model, sched, cond, uncond, args.size, device, seed=args.seed)
        else:
            with torch.no_grad():
                vtok = img_proj(vision(pixel_values=ref_px).last_hidden_state).repeat_interleave(args.n, 0)
            imgs = sample(model, sched, torch.cat([cond, vtok], 1),
                          torch.cat([uncond, vtok], 1), args.size, device, seed=args.seed)

        gen = torch.stack([clip_view(to_rgba(im)) for im in imgs]).to(device)
        with torch.no_grad():
            g = F.normalize(clip.get_image_features(pixel_values=gen), dim=-1)
        g = g.view(len(texts), args.n, -1)
        ref_sim = (g * ref_feat[:, None, :]).sum(-1)     # per ref, per sample
        txt_sim = (g * txt_feat[:, None, :]).sum(-1)
        # calibration floor: how similar is a reference to the WRONG sprite?
        # without this, an absolute ref_sim of 0.62 means nothing.
        all_pairs = torch.einsum("rd,snd->rsn", ref_feat, g)          # ref r vs samples of prompt s
        eye = torch.eye(len(texts), device=all_pairs.device, dtype=torch.bool)
        mismatch = all_pairs[~eye].mean()
        gap = ref_sim.mean() - mismatch
        name = Path(ckpt).stem
        rows.append((name, ref_sim.mean().item(), txt_sim.mean().item(),
                     mismatch.item(), gap.item()))
        print(f"{name}: ref_sim={ref_sim.mean():.4f} mismatch={mismatch:.4f} "
              f"GAP={gap:.4f} txt_sim={txt_sim.mean():.4f}", flush=True)
        for i, q in enumerate(ref_paths):
            print(f"    {q.stem[:28]:30s} ref={ref_sim[i].mean():.3f} txt={txt_sim[i].mean():.3f}",
                  flush=True)
        del model, img_proj
        torch.cuda.empty_cache()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "ref_sim", "txt_sim", "mismatch", "gap"])
        w.writerows(rows)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
