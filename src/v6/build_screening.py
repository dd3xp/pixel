"""Screening pairs for the human study: two real sprites, one drawn at 16 px and one shrunk to it.

The model judge we tried ranked large generators above real sprites when asked which looked more like
real pixel art, so it could not be trusted about ours. Human raters may do the same. These pairs have a
ground truth that does not depend on anyone's taste -- one image was authored at this size and the other
was not -- so a rater's answers on them measure whether they can see the distinction at all, and the
judgement trials of raters who cannot are reported separately.

Both sides go through the same to_tensor the training pipeline uses, so they differ in provenance and
not in processing.

Usage: python src/v6/build_screening.py --n 20 --out runs_out/human_study2
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from fd_fair import native_sizes  # noqa: E402
from train_cond import to_tensor  # noqa: E402

CHECK = ((235, 235, 235), (205, 205, 205))


def board(im, scale=16, cell=256):
    im = im.convert("RGBA")
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    bg = Image.new("RGB", im.size)
    px = bg.load()
    for y in range(bg.height):
        for x in range(bg.width):
            px[x, y] = CHECK[((x // 8) + (y // 8)) % 2]
    bg.paste(im, (0, 0), im)
    return bg.resize((cell, cell), Image.NEAREST) if bg.size != (cell, cell) else bg


def at_R(path, R):
    t = to_tensor(Image.open(path).convert("RGBA"), R)
    a = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a, "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", default="runs_out/human_study2")
    a = ap.parse_args()
    R = a.size
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)

    real = sorted({p.replace("\\", "/") for p in
                   glob.glob("data/oga_clean/**/*.png", recursive=True) + glob.glob("data/oga_clean/*.png")})
    sz = native_sizes(real)
    native = [p for p in real if sz[p] <= R]
    shrunk = [p for p in real if 25 <= sz[p] <= 64]
    rng = random.Random(7)
    rng.shuffle(native)
    rng.shuffle(shrunk)
    print(f"{len(native)} natively <= {R}px, {len(shrunk)} at 25-64px", flush=True)

    pairs = []
    for i in range(a.n):
        pn, ps = native[i], shrunk[i]
        board(at_R(pn, R)).save(out / "img" / f"screen{i:02d}_native.png")
        board(at_R(ps, R)).save(out / "img" / f"screen{i:02d}_shrunk.png")
        left_is_native = rng.random() < 0.5
        pairs.append({
            "id": f"screen{i:02d}",
            "left_img": f"img/screen{i:02d}_{'native' if left_is_native else 'shrunk'}.png",
            "right_img": f"img/screen{i:02d}_{'shrunk' if left_is_native else 'native'}.png",
            "answer": "left" if left_is_native else "right",
        })
    (out / "screening.json").write_text(json.dumps({"size": R, "pairs": pairs}, indent=1), encoding="utf-8")
    print(f"wrote {out}/screening.json with {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
