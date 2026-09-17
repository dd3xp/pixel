"""Build an evaluation prompt set whose ground truth is real *native* pixel art.

Until now the held-out prompts (and therefore the qualitative figures' ground-truth row) came from the mixed corpus,
where two thirds of the sprites are 25-64 px artwork squashed to 16 px: the reference images themselves look soft, so a
figure that shows them is not evidence about pixel art.  This selects the held-out sprites that are natively at most R
px -- the same pool the native FD protocol scores against -- and writes their captions plus the matching sprite images.

Usage: python src/v6/native_prompts.py --size 16 --n 600
Writes runs_out/heldout_native<R>_prompts.txt and runs_out/heldout_native<R>_gt/<i>_0.png
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import fd_fair as FF  # noqa: E402
from train_cond import to_tensor  # noqa: E402
from sample_cond import to_rgba  # noqa: E402

CAPS = [("data/oga_clean", "data/oga_captions_recap.csv"),
        ("data/extra_all", "data/extra_all_recap.csv"),
        ("data/oga_clean", "data/tool_candidates_recap.csv"),
        ("data/oga3_clean", "data/oga3_captions_recap.csv")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--pool", default="union", choices=["old", "union"])
    a = ap.parse_args()
    cap = {}
    for d, f in CAPS:
        if not Path(f).exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            cap[f"{d}/{r['path']}".replace("\\", "/")] = r["text"]
    # The native split of fd_fair is NOT the training hold-out: a prompt has to be in runs_out/holdout_exclude.txt
    # (the list the training run actually excluded), or the model has seen its sprite and reconstructs it (09-18).
    excl = {l.strip().replace("\\", "/") for l in open("runs_out/holdout_exclude.txt", encoding="utf-8") if l.strip()}
    _, held = FF.real_split(a.size, a.pool)
    rows = [(p, cap[p]) for p in held
            if p in excl and p in cap and len(cap[p].split()) >= 4 and "unclear" not in cap[p].lower()]
    rows = rows[:a.n]
    outp = Path(f"runs_out/heldout_native{a.size}_prompts.txt")
    outp.write_text("\n".join(t for _, t in rows) + "\n", encoding="utf-8")
    gt = Path(f"runs_out/heldout_native{a.size}_gt")
    gt.mkdir(parents=True, exist_ok=True)
    for i, (p, _) in enumerate(rows):
        to_rgba((to_tensor(Image.open(p).convert("RGBA"), a.size) + 1) / 2).save(gt / f"{i}_0.png")
    print(f"{len(rows)} native <= {a.size}px held-out prompts -> {outp} (ground truth in {gt})", flush=True)


if __name__ == "__main__":
    main()
