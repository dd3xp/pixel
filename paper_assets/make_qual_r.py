"""Qualitative figures at every resolution, prompt-aligned: column c = held-out caption c (seed 0) in every row.
Inputs: paper_assets/qual_r/<dir>/s<R>/<c:02d>_0.png (samples) and qual_r/heldout3000_totensor_s<R>/<c:05d>.png (real),
pulled with baseline/pull_qual.sh.  Nearest-upsampled to a fixed 128 px cell on a light checkerboard.
Run from repo root: python paper_assets/make_qual_r.py   -> paper_assets/fig_qual_{12,16,20,24,32}px.png
(supersedes make_qual.py's fig_qual_16px.png, whose real row was misaligned beyond column 10: sorted-order pull).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Q = Path("paper_assets/qual_r")
E = "_matched_eval"
FIGS = {
    12: [("real (held-out)", None), ("CFG w=4  (12.91)", "v7h_r12_cfg4"), ("best CFG w=2  (9.71)", "v7h_r12_cfg2"), ("autoguidance 10k w=1.5  (9.18)", "v7h_r12_autog10k_w1p5"),
         ("reverse: bucket:16 ref w=2  (14.45)", "v7h_r12_bk16_w2")],
    16: [("real (held-out)", None), ("CFG w=4  (21.98)", "v7h"), ("best CFG w=1.5  (12.24)", "v7h_cfg1p5"), ("autoguidance 10k w=1.5  (8.98)", "v7h_autog10k_w1p5"),
         ("bucket:12 self-guidance w=2  (8.59)", "v7h_gbk12_w2"), ("composed: 10k under bucket:12 w=1.5  (7.67)", "v7h_gbk12s10k_w1p5"), ("composed o bucketu:12 w=1.5  (8.60)", "v7h_bku12s10k_w1p5"),
         ("weak ref alone: bucket:12 belief  (36.43)", "v7h_bk12only"), ("reverse: bucket:24 ref w=2  (17.27)", "v7h_gbk24_w2")],
    20: [("real (held-out)", None), ("CFG w=4  (45.92)", "v7h_r20_cfg4"), ("best CFG w=2.5  (39.53)", "v7h_r20_cfg2p5"), ("autoguidance 10k w=1.5  (31.78)", "v7h_r20_autog10k_w1p5"),
         ("bucket:16 self-guidance w=2  (32.89)", "v7h_r20_bk16_w2"), ("composed: 10k under bucket:16 w=1.5  (29.66)", "v7h_r20_bk16s10k_w1p5"), ("composed o bucketu:16 w=1.5  (29.81)", "v7h_r20_bku16s10k_w1p5"),
         ("weak ref alone: bucket:16 belief  (94.02)", "v7h_r20_bk16only"), ("reverse: bucket:32 ref w=2  (59.13)", "v7h_r20_bk32_w2")],
    24: [("real (held-out)", None), ("CFG w=4  (79.80)", "v7h_r24_cfg4"), ("best CFG w=2  (67.34)", "v7h_r24_cfg2"), ("autoguidance 10k w=1.5  (53.36)", "v7h_r24_autog10k_w1p5"),
         ("bucket:16 self-guidance w=2  (60.41)", "v7h_r24_bk16_w2"), ("composed: 10k under bucket:16 w=1.5  (48.70)", "v7h_r24_bk16s10k_w1p5"), ("composed o bucketu:16 w=1.5  (49.78)", "v7h_r24_bku16s10k_w1p5"),
         ("weak ref alone: bucket:16 belief  (173.39)", "v7h_r24_bk16only"), ("reverse: bucket:32 ref w=2  (101.67)", "v7h_r24_bk32_w2")],
    32: [("real (held-out)", None), ("CFG w=4  (96.85)", "v7h_r32_cfg4"), ("best CFG w=2  (83.65)", "v7h_r32_cfg2"), ("autoguidance 10k w=1.5  (75.23)", "v7h_r32_autog10k_w1p5"),
         ("bucket:24 self-guidance w=2  (77.45)", "v7h_r32_bk24_w2"), ("composed: 10k under bucket:24 w=1.5  (69.27)", "v7h_r32_bk24s10k_w1p5"), ("composed o bucketu:24 w=1.5  (65.27)", "v7h_r32_bku24s10k_w1p5"),
         ("reverse: bucket:48 ref w=2  (113.93)", "v7h_r32_bk48_w2")],
}
NCOL, CELL, PAD, LABEL_W = 20, 128, 4, 400


def load(R, d, c):
    f = Q / f"heldout3000_totensor_s{R}" / f"{c:05d}.png" if d is None else Q / (d + E) / f"s{R}" / f"{c:02d}_0.png"
    im = Image.open(f).convert("RGBA")
    assert im.size == (R, R), (f, im.size)
    return im


def checker(w, h, k=8):
    a = np.indices((h, w)).sum(0) // k % 2
    return Image.fromarray((235 + 12 * a).astype(np.uint8)).convert("RGBA")


def main():
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for R, rows in FIGS.items():
        W = LABEL_W + NCOL * (CELL + PAD) + PAD
        H = len(rows) * (CELL + PAD) + PAD
        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for r, (label, d) in enumerate(rows):
            y = PAD + r * (CELL + PAD)
            draw.text((8, y + CELL // 2 - 10), label, fill=(0, 0, 0, 255), font=font)
            for c in range(NCOL):
                x = LABEL_W + PAD + c * (CELL + PAD)
                bg = checker(CELL, CELL)
                bg.alpha_composite(load(R, d, c).resize((CELL, CELL), Image.NEAREST))
                canvas.paste(bg, (x, y))
        out = f"paper_assets/fig_qual_{R}px.png"
        canvas.convert("RGB").save(out)
        print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
