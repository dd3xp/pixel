"""Qualitative figure: same seed (0) + same held-out captions, 16 px, v7h weights.
Rows: real held-out / v7h CFG w=4 / autoguidance (step10k) w=1.5 / bucket:12 self-guidance w=2 / stack w=1.5 /
pure bucket:12 belief (the weak reference itself).  Nearest-upsampled x8 on a light checkerboard.
Also: weak-reference TV vs guided FD scatter (dmech table).  Run from repo root: python paper_assets/make_qual.py
"""
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Q = Path("paper_assets/qual")
ROWS = [
    ("real (held-out)", "real16"),
    ("v7h  CFG w=4  (21.98)", "v7h_matched_eval"),
    ("autoguidance 10k  w=1.5  (8.98)", "v7h_autog10k_w1p5_matched_eval"),
    ("bucket:12 self-guidance  w=2  (8.59)", "v7h_gbk12_w2_matched_eval"),
    ("stack: bucket:12 + 10k  w=1.5  (7.67)", "v7h_gbk12s10k_w1p5_matched_eval"),
    ("weak ref alone: bucket:12 belief  (36.43)", "v7h_bk12only_matched_eval"),
]
IDX = list(range(24))
S, PAD, LABEL_W = 8, 4, 300


def load(d, i):
    files = sorted(glob.glob(str(Q / d / "*.png")))
    im = Image.open(files[i]).convert("RGBA")
    assert im.size == (16, 16), (d, im.size)
    return im


def checker(w, h, k=4):
    a = np.indices((h, w)).sum(0) // k % 2
    return Image.fromarray((235 + 12 * a).astype(np.uint8)).convert("RGBA")


def main():
    cell = 16 * S
    W = LABEL_W + len(IDX) * (cell + PAD) + PAD
    H = len(ROWS) * (cell + PAD) + PAD
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for r, (label, d) in enumerate(ROWS):
        y = PAD + r * (cell + PAD)
        draw.text((8, y + cell // 2 - 10), label, fill=(0, 0, 0, 255), font=font)
        for c, i in enumerate(IDX):
            x = LABEL_W + PAD + c * (cell + PAD)
            bg = checker(cell, cell)
            im = load(d, i).resize((cell, cell), Image.NEAREST)
            bg.alpha_composite(im)
            canvas.paste(bg, (x, y))
    canvas.convert("RGB").save("paper_assets/fig_qual_16px.png")
    print("wrote paper_assets/fig_qual_16px.png", canvas.size)

    # weak-reference TV vs FD after guidance (dmech, experiment_log 09-07 14:20); strong model TV = 32.0
    pts = {  # name: (TV of pure weak reference, FD of guided result with that reference, marker)
        "bucket:12 (w=2)": (21.6, 8.59), "snapshot 10k (w=1.5)": (27.3, 8.98), "unconditional (CFG w=4)": (27.3, 21.98),
        "bucket:24 (w=2)": (31.9, 17.27), "bucket:64 (w=2)": (40.9, 19.34),
        "2x2 block-avg branch, trained (w=1.5)": (14.2, 21.53), "2x2 block-avg branch (w=2)": (14.2, 35.22),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("no matplotlib; skip scatter")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (tv, fd) in pts.items():
        aligned = "block" not in name
        ax.scatter(tv, fd, c="C0" if aligned else "C3", marker="o" if aligned else "x", s=60)
        ax.annotate(name, (tv, fd), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axvline(32.0, ls="--", c="gray", lw=1)
    ax.text(32.3, 33, "strong model TV = 32.0", fontsize=7, color="gray")
    ax.axhline(21.98, ls=":", c="gray", lw=1)
    ax.text(15, 22.6, "bare v7h CFG w=4 = 21.98", fontsize=7, color="gray")
    ax.set_xlabel("TV of the pure weak reference (mean |dRGB|, opaque neighbours)")
    ax.set_ylabel("FD-DINOv2 @16 px after guidance")
    ax.set_title("Structure-aligned references (o) help iff TV < strong model; misaligned (x) hurt regardless", fontsize=8)
    fig.tight_layout()
    fig.savefig("paper_assets/fig_tv_vs_fd.png", dpi=200)
    print("wrote paper_assets/fig_tv_vs_fd.png")


if __name__ == "__main__":
    main()
