"""How much can the text-only model already draw?

Everything we tried for "draw anything" assumed the text path is limited to the
training corpus and that image conditioning is the only way out.  That was
inferred from a handful of out-of-domain references, never measured.  The corpus
is 43,851 game sprites and does contain characters, furniture, vehicles, plants
-- so the real vocabulary may be much wider than assumed, and the answer decides
whether a large modelling effort is warranted at all.

Protocol: 100 everyday objects, best-of-8 by CLIP (the same reranking the
baselines get), one grid per batch to score by eye, plus the CLIP score itself
as a coarse automatic signal.  CLIP score is NOT a quality measure -- it is
reported per prompt so the eyeball scoring has something to sort by, and so a
later run can be compared against this one on identical prompts and seeds.

Usage: CUDA_VISIBLE_DEVICES=6 python src/v6/coverage.py --ckpt workdir/v7c_bow/model_latest.pt \
           --size 16 --out runs_out/coverage16
"""
import argparse
import csv
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

OBJECTS = [
    # in-domain controls: these must stay good, or the run says nothing
    "golden sword", "iron pickaxe", "health potion", "gold coin", "red apple",
    # tools and household
    "smartphone", "laptop computer", "coffee machine", "umbrella", "wristwatch",
    "camera", "light bulb", "scissors", "hammer", "screwdriver",
    "padlock", "key", "compass", "telescope", "microscope",
    "toothbrush", "hairbrush", "mirror", "candle", "lantern",
    "teapot", "coffee cup", "wine glass", "fork", "spoon",
    # food
    "pizza slice", "hamburger", "hot dog", "ice cream cone", "birthday cake",
    "banana", "watermelon slice", "bunch of grapes", "carrot", "mushroom",
    "loaf of bread", "cheese wedge", "fried egg", "sushi roll", "donut",
    # nature
    "cactus", "sunflower", "oak tree", "autumn leaf", "seashell",
    "starfish", "snowflake", "lightning bolt", "rainbow", "campfire",
    # animals
    "rubber duck", "black cat", "goldfish", "butterfly", "owl",
    "penguin", "snail", "crab", "bee", "frog",
    # vehicles
    "race car", "bicycle", "sailboat", "rocket ship", "hot air balloon",
    "helicopter", "train engine", "skateboard", "tractor", "submarine",
    # music and play
    "acoustic guitar", "drum", "trumpet", "piano keyboard", "violin",
    "soccer ball", "basketball", "chess piece", "playing card", "dice",
    # clothing and misc
    "top hat", "sneaker", "backpack", "sunglasses", "necktie",
    "treasure chest", "spell book", "hourglass", "anchor", "crown",
    "magnet", "battery", "traffic cone", "fire extinguisher", "mailbox",
]


def on_white(im, side=224):
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB").resize((side, side), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "workdir/v7c_bow/model_latest.pt"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=8, help="candidates per prompt")
    ap.add_argument("--batch", type=int, default=10, help="prompts per forward pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    out = Path(args.out)
    (out / "best").mkdir(parents=True, exist_ok=True)

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = build_model(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["unet"] if isinstance(ck, dict) and "unet" in ck else ck)
    model.eval()
    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

    rows, chosen = [], []
    for start in range(0, len(OBJECTS), args.batch):
        chunk = OBJECTS[start:start + args.batch]
        prompts = [f"a pixel art {o}" for o in chunk]
        cond = embed_text(prompts, tok, enc, device).repeat_interleave(args.n, 0)
        uncond = embed_text([""] * len(chunk) * args.n, tok, enc, device)
        imgs = [to_rgba(x) for x in
                sample(model, sched, cond, uncond, args.size, device, seed=args.seed)]
        with torch.no_grad():
            t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
            f_t = F.normalize(clip.get_text_features(**t), dim=-1)
        for j, obj in enumerate(chunk):
            group = imgs[j * args.n:(j + 1) * args.n]
            with torch.no_grad():
                px = proc(images=[on_white(g) for g in group], return_tensors="pt").to(device)
                f_i = F.normalize(clip.get_image_features(**px), dim=-1)
            scores = (f_i @ f_t[j])
            k = int(scores.argmax())
            group[k].save(out / "best" / f"{start + j:03d}_{obj.replace(' ', '_')}.png")
            chosen.append((obj, group[k]))
            rows.append((start + j, obj, float(scores[k]), float(scores.mean())))
        print(f"[{start + len(chunk)}/{len(OBJECTS)}]", flush=True)

    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "object", "clip_best", "clip_mean"])
        w.writerows(rows)

    # contact sheets of 25 so they stay readable when scored by eye
    cell, pad = 96, 16
    for page in range(0, len(chosen), 25):
        part = chosen[page:page + 25]
        cols = 5
        rws = (len(part) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rws * (cell + pad)), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for i, (obj, im) in enumerate(part):
            r, c = divmod(i, cols)
            y = r * (cell + pad)
            bg = Image.new("RGBA", (cell, cell), (255, 255, 255, 255))
            sheet.paste(Image.alpha_composite(
                bg, im.resize((cell, cell), Image.NEAREST)).convert("RGB"), (c * cell, y))
            d.text((c * cell + 3, y + cell + 2), obj[:20], fill=(0, 0, 0))
        sheet.save(out / f"sheet_{page // 25}.png")
    print(f"-> {out}  ({len(rows)} objects, best of {args.n})", flush=True)


if __name__ == "__main__":
    main()
