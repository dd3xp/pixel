"""Assemble the forced-choice study as the web page consumes it.

build_human_study.py produced the first version; this one is the package that actually goes to raters,
and differs in three ways decided after the model-judge run:

  * screening trials are included, twenty pairs of two real sprites -- one authored at 16 px, one larger
    artwork reduced to it -- whose answer is a fact about the file. A rater who cannot separate those
    cannot be read as evidence about ours, which is exactly how the vision--language judge failed;
  * SDXL is dropped. It loses to everything under every measurement we have, so asking people about it
    spends their attention without buying an answer;
  * ours is the k=6 configuration, the one the paper recommends because it reproduces the medium's
    colour statistics (six colours, 0.43 flat neighbours against six and 0.46 for real sprites). k=4 has
    the better FD and is flatter than the real thing.

Prompts are drawn with a fixed seed and nothing is hand-picked: the point of the study is to answer the
objection that our reference set is our own, and a curated sample would answer it in the wrong
direction.

09-25: two faults in the first package, both found by measuring the images that actually shipped.
  * the palettes were not matched. ours was icg2_pal6 while real, gpt and flux all came from the _q4
    directories, so our samples carried 5.3 colours on average against 3.5 for the real sprites and
    3.3 / 3.7 for the two baselines -- a 50% larger palette for our own system than for everything it
    was compared against, the real reference included. --systems now sets the directories, and the
    rerun puts every arm on the same k=6 projection;
  * prompt 00122 rendered fully transparent for both real and ours, so any pair against gpt or flux on
    that prompt was decided by one side being blank. --drop-blank rejects a prompt when any arm has no
    opaque pixel.

Usage:
  python src/v6/build_study_web.py --n 24 --out runs_out/study_web
  python src/v6/build_study_web.py --n 24 --drop-blank       --systems real=runs_out/ext200/q/real_q6/s16 ours=runs_out/ext200/v8n/icg2_pal6/s16                 gpt=runs_out/ext200/q/gpt_q6/s16 flux=runs_out/ext200/q/flux2_q6/s16
"""
import argparse
import itertools
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SYSTEMS = {
    "real": "runs_out/ext200/q/real_q4/s16",
    "ours": "runs_out/ext200/v8n/icg2_pal6/s16",
    "gpt":  "runs_out/ext200/q/gpt_q4/s16",
    "flux": "runs_out/ext200/q/flux2_q4/s16",
}
QUESTIONS = {
    "fidelity": "Which one looks like it was drawn pixel by pixel at this size, rather than a larger picture shrunk down?",
    "preference": "Which one would you rather put in a game?",
}
CHECK = ((235, 235, 235), (205, 205, 205))
SCALE, CELL = 16, 256


def board(im):
    im = im.convert("RGBA")
    im = im.resize((im.width * SCALE, im.height * SCALE), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 8) + (y // 8)) % 2]
    bg.paste(im, (0, 0), im)
    return bg.resize((CELL, CELL), Image.NEAREST) if bg.size != (CELL, CELL) else bg


def by_index(d):
    out = {}
    for f in Path(d).glob("*.png"):
        head = f.stem.split("_")[0]
        if head.isdigit():
            out.setdefault(int(head), f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="prompts drawn from the 200")
    ap.add_argument("--out", default="runs_out/study_web")
    ap.add_argument("--prompts", default="runs_out/ext200/v8n/icg2_pal4/prompts.txt")
    ap.add_argument("--screening", default="runs_out/human_study2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", nargs="*", default=[], help="name=dir, overriding SYSTEMS")
    ap.add_argument("--drop-blank", action="store_true",
                    help="reject a prompt when any arm renders with no opaque pixel")
    a = ap.parse_args()
    systems = dict(s.split("=", 1) for s in a.systems) if a.systems else dict(SYSTEMS)
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    by_sys = {}
    for k, d in systems.items():
        got = by_index(d)
        if not got:
            raise SystemExit(f"{k}: no indexable images under {d}")
        print(f"{k:5s} {len(got):4d} prompts  {d}")
        by_sys[k] = got
    idxs = sorted(set.intersection(*(set(v) for v in by_sys.values())))
    if a.drop_blank:
        def blank(p):
            im = Image.open(p).convert("RGBA")
            return not (np.array(im)[:, :, 3] > 127).any()
        bad = sorted(i for i in idxs if any(blank(by_sys[k][i]) for k in by_sys))
        idxs = [i for i in idxs if i not in bad]
        print(f"dropped {len(bad)} prompts with a blank arm: {bad}")
    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    picked = sorted(rng.sample(idxs, a.n))

    for k, got in by_sys.items():
        for i in picked:
            board(Image.open(got[i])).save(out / "img" / f"{k}_{i:05d}.png")

    trials = []
    for i in picked:
        for x, y in itertools.combinations(systems, 2):
            if "real" not in (x, y) and "ours" not in (x, y):
                continue          # external vs external decides nothing in the paper
            left, right = (x, y) if rng.random() < 0.5 else (y, x)
            for q, qt in QUESTIONS.items():
                trials.append({"id": f"{q}-{i:05d}-{left}-{right}", "kind": "judge",
                               "index": i, "prompt": prompts[i] if i < len(prompts) else "",
                               "question": q, "text": qt, "left": left, "right": right,
                               "left_img": f"img/{left}_{i:05d}.png",
                               "right_img": f"img/{right}_{i:05d}.png"})

    scr = json.loads((Path(a.screening) / "screening.json").read_text(encoding="utf-8"))
    screening = []
    for p in scr["pairs"]:
        for side in ("left_img", "right_img"):
            src = Path(a.screening) / p[side]
            shutil.copy2(src, out / "img" / src.name)
        screening.append({"id": p["id"], "kind": "screen",
                          "question": "fidelity", "text": QUESTIONS["fidelity"],
                          "left_img": "img/" + Path(p["left_img"]).name,
                          "right_img": "img/" + Path(p["right_img"]).name,
                          "answer": p["answer"]})

    (out / "study.json").write_text(json.dumps({
        "systems": list(systems), "dirs": systems, "questions": QUESTIONS, "scale": SCALE, "cell": CELL,
        "n_prompts": a.n, "seed": a.seed, "judge": trials, "screen": screening}, indent=1),
        encoding="utf-8")
    print(f"{len(trials)} judgement trials ({a.n} prompts x {len(trials) // (a.n * len(QUESTIONS))} pairs "
          f"x {len(QUESTIONS)} questions) + {len(screening)} screening pairs")
    print(f"wrote {out}/study.json and {len(list((out / 'img').glob('*.png')))} images")


if __name__ == "__main__":
    main()
