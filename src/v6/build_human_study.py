"""Assemble the material for the two-alternative forced-choice study (CVPR round, 09-20).

The paper's central claim -- that our samples are closer to real low-resolution pixel art than a large
generator's downscaled output -- currently rests on a reference set we defined ourselves.  A reviewer is
entitled to call that self-serving.  The answer is to ask people, with two questions that we expect to
disagree:

  fidelity:   "Which one looks more like a real game sprite drawn at this size?"
  preference: "Which one would you rather put in a game?"

Systems compared at 16 px, all already generated on the same 200 external-comparison prompts:
  real  q/real_q4/s16          held-out sprites, same palette post-process
  ours  v8n/icg2_pal4/s16      ICG w2 + 4-colour projection
  flux  q/flux2_q4/s16         FLUX.2-klein + pixel-art LoRA, downscaled
  gpt   q/gpt_q4/s16           gpt-image-2, downscaled
  sdxl  q/lora_q4/s16          SDXL + Pixel Art XL, downscaled

Writes <out>/img/<system>_<idx>.png (nearest-neighbour upscale, checkerboard alpha) and <out>/trials.json
with one record per trial: prompt, the two systems, the two image paths, and both questions.  Nothing is
published and no rater is contacted: this only checks that the material exists and is aligned.

Usage: python src/v6/build_human_study.py --n 40 --out runs_out/human_study
"""
import argparse
import itertools
import json
import random
from pathlib import Path

from PIL import Image

SYSTEMS = {
    "real": "runs_out/ext200/q/real_q4/s16",
    "ours": "runs_out/ext200/v8n/icg2_pal4/s16",
    "flux": "runs_out/ext200/q/flux2_q4/s16",
    "gpt":  "runs_out/ext200/q/gpt_q4/s16",
    "sdxl": "runs_out/ext200/q/lora_q4/s16",
}
QUESTIONS = {
    "fidelity": "Which one looks more like a real game sprite drawn at this size?",
    "preference": "Which one would you rather put in a game?",
}
SCALE = 8
CHECK = ((235, 235, 235), (205, 205, 205))


def on_checkerboard(im, scale=SCALE, cell=4):
    im = im.convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // cell) + (y // cell)) % 2]
    bg.paste(im, (0, 0), im)
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="prompts sampled from the 200")
    ap.add_argument("--out", default="runs_out/human_study")
    ap.add_argument("--prompts", default="runs_out/ext200/v8n/icg2_pal4/prompts.txt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # the systems do not agree on file names -- our sampler writes "<prompt>_<seed>.png", the converted
    # baselines write "00000.png" -- so index every system by the prompt ordinal in its name instead
    by_sys = {}
    for k, d in SYSTEMS.items():
        got = {}
        for f in Path(d).glob("*.png"):
            head = f.stem.split("_")[0]
            if head.isdigit():
                got.setdefault(int(head), f)     # first seed only
        if not got:
            raise SystemExit(f"{k}: no indexable images under {d}")
        print(f"{k:5s} {len(got):4d} prompts  {d}")
        by_sys[k] = got
    idxs = sorted(set.intersection(*(set(v) for v in by_sys.values())))
    print(f"prompt indices common to every system: {len(idxs)}")
    if len(idxs) < a.n:
        raise SystemExit(f"only {len(idxs)} shared prompts, asked for {a.n}")

    prompts = []
    if Path(a.prompts).exists():
        prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
        print(f"prompts file: {len(prompts)} lines")

    rng = random.Random(a.seed)
    picked = rng.sample(idxs, a.n)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    for k, got in by_sys.items():
        for i in picked:
            on_checkerboard(Image.open(got[i])).save(out / "img" / f"{k}_{i:05d}.png")

    trials = []
    for i in picked:
        text = prompts[i] if i < len(prompts) else ""
        for x, y in itertools.combinations(SYSTEMS, 2):
            if "real" not in (x, y) and "ours" not in (x, y):
                continue          # skip external-vs-external: nothing in the paper turns on it
            left, right = (x, y) if rng.random() < 0.5 else (y, x)
            for q, qt in QUESTIONS.items():
                trials.append({"index": i, "prompt": text, "question": q, "text": qt,
                               "left": left, "right": right,
                               "left_img": f"img/{left}_{i:05d}.png",
                               "right_img": f"img/{right}_{i:05d}.png"})
    rng.shuffle(trials)
    json.dump({"systems": list(SYSTEMS), "questions": QUESTIONS, "scale": SCALE,
               "n_prompts": a.n, "trials": trials},
              open(out / "trials.json", "w", encoding="utf-8"), indent=1)
    pairs = len(trials) // len(QUESTIONS) // a.n
    print(f"{len(trials)} trials = {a.n} prompts x {pairs} pairs x {len(QUESTIONS)} questions")
    print(f"wrote {out}/trials.json and {len(picked) * len(SYSTEMS)} images")


if __name__ == "__main__":
    main()
