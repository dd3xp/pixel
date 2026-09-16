"""Qualitative figures for the ICLR 2027 draft (16 px, seed 0, same prompt per column).

Adapted from paper_assets/make_qual.py: nearest-neighbour upsampling of the native 16x16 RGBA sprite onto a light
checkerboard (so transparency is visible). Run from the repository root:
    python paper_assets/iclr27/figures/make_figs.py

fig_qual16    rows = methods on the recaptioned model v7r (matched protocol samples, prompt index = column).
fig_external16 rows = external "generate large, then shrink" systems vs ours (first 200 held-out prompts, n = 200 protocol).

The "+ downscale" sprites are re-derived from the saved 1024 px renders with verbatim copies of the project's own
cutout (baseline/api_to_sprites.py) and to_tensor (src/v6/train_v7.py); the FD numbers in the labels come from the
experiment log, not from these files. Columns were picked from the first 60 held-out prompts for legibility of the
real sprite; every row uses the same columns, so there is no per-method selection.
"""
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
R = ROOT / "runs_out/server_node03/runs_out"
OUT = ROOT / "paper_assets/iclr27/figures"
COLS = [1, 4, 5, 9, 10, 31, 42, 44, 55, 58]
S = 8            # upsampling factor: 16 px -> 128 px cells
PAD = 6
LABEL_W = 430
FONT_PT = 36


def font(bold=False):
    for name in (("timesbd.ttf" if bold else "times.ttf"), "DejaVuSerif.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, FONT_PT)
        except OSError:
            continue
    return ImageFont.load_default()


def checker(n, k=16):
    yy, xx = np.indices((n, n))
    a = (yy // k + xx // k) % 2
    return Image.fromarray((232 + 16 * a).astype(np.uint8)).convert("RGBA")


# ---- verbatim logic of baseline/api_to_sprites.cutout and src/v6/train_v7.to_tensor (numpy/PIL only) ----
def cutout(img, tol=18):
    rgb = np.asarray(img.convert("RGB"))
    a = rgb.astype(np.int16)
    h, w, _ = a.shape
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    close = np.abs(a - bg).max(axis=2) <= tol
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if close[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    alpha = (~seen).astype(np.uint8) * 255
    out = np.dstack([rgb, alpha])
    ys, xs = np.where(alpha > 0)
    if len(ys) == 0:
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


def downscale_rgba(im, side):
    f = side / max(im.size)
    w, h = max(1, round(im.width * f)), max(1, round(im.height * f))
    a = np.array(im).astype(np.float32)
    a[:, :, :3] *= a[:, :, 3:4] / 255.0
    pm = Image.fromarray(a.clip(0, 255).astype(np.uint8), "RGBA").resize((w, h), Image.BOX)
    b = np.array(pm).astype(np.float32)
    alpha = b[:, :, 3:4]
    b[:, :, :3] = np.where(alpha > 0, b[:, :, :3] / np.maximum(alpha, 1) * 255.0, 0)
    return Image.fromarray(b.clip(0, 255).astype(np.uint8), "RGBA")


def to_sprite(im, side):
    if max(im.size) > side:
        im = downscale_rgba(im, side)
    elif max(im.size) * 2 <= side:
        f = side // max(im.size)
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    a = np.array(canvas).astype(np.float32)
    a[a[:, :, 3] < 128] = 0.0
    a[:, :, 3] = (a[:, :, 3] >= 128) * 255.0
    return Image.fromarray(a.astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------------------------------------------
def matched(tag):
    return lambda i: Image.open(R / f"{tag}_matched_eval/s16/{i:02d}_0.png").convert("RGBA")


def real(i):
    return Image.open(R / f"heldout3000_totensor_s16/{i:05d}.png").convert("RGBA")


def ext_small(d):
    return lambda i: Image.open(ROOT / f"runs/{d}/s16/{i:05d}.png").convert("RGBA")


def ext_down(model):
    return lambda i: to_sprite(cutout(Image.open(ROOT / f"runs/{model}/big/{i:05d}.png")), 16)


def ext_big(model):   # the 1024 px render, shown only for reference (bicubic display thumbnail)
    return lambda i: Image.open(ROOT / f"runs/{model}/big/{i:05d}.png").convert("RGBA").resize((16 * S, 16 * S),
                                                                                             Image.LANCZOS)


def render(rows, name, sep_after=()):
    cell = 16 * S
    W = LABEL_W + len(COLS) * (cell + PAD) + PAD
    extra = 14 * len(sep_after)
    H = len(rows) * (cell + PAD) + PAD + extra
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = PAD
    for r, (label, fn, bold) in enumerate(rows):
        f = font(bold)
        lines = label.split("\n")
        ty = y + cell // 2 - (len(lines) * (FONT_PT + 4)) // 2
        for ln in lines:
            draw.text((10, ty), ln, fill=(0, 0, 0, 255), font=f)
            ty += FONT_PT + 4
        for c, i in enumerate(COLS):
            x = LABEL_W + PAD + c * (cell + PAD)
            im = fn(i)
            bg = checker(cell)
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.NEAREST)
            bg.alpha_composite(im)
            canvas.paste(bg, (x, y))
        y += cell + PAD
        if r in sep_after:
            draw.line([(10, y + 4), (W - 10, y + 4)], fill=(120, 120, 120, 255), width=2)
            y += 14
    rgb = canvas.convert("RGB")
    rgb.save(OUT / f"{name}.png")
    rgb.save(OUT / f"{name}.pdf", resolution=300)
    print("wrote", OUT / f"{name}.png", rgb.size)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qual = [
        ("Real held-out
sprite", real, False),
        ("CFG, best w = 1.5
FD 11.28", matched("v7r_cfg1p5"), False),
        ("CDG, w = 1.5
FD 10.36 (seed 0)", matched("v7r_cdg_w1p5"), False),
        ("ICG, w = 1.5
FD 7.46", matched("v7r_icg_w1p5"), False),
        ("Label ref., w = 2
FD 8.86", matched("v7r_bk12_w2"), False),
        ("Composed ref., w = 1.5
FD 8.50", matched("v7r_stk12_w1p5"), False),
        ("GFT (empty caption)
1 NFE, FD 10.05", matched("v7r_gftnull_w1p5"), False),
        ("Ours, internalized
1 NFE, FD 7.23", matched("v7r_gft_w1p5"), True),
        ("Ours + ICG
2 NFE, FD 6.25", matched("v7r_gft_w1p25_icg1p5"), True),
    ]
    render(qual, "fig_qual16", sep_after=(0, 6))
    ext = [
        ("Real held-out
sprite (26.53)", real, False),
        ("gpt-image-2
1024 px render", ext_big("ext_recap_gpt"), False),
        ("  + downscale
  (82.23)", ext_down("ext_recap_gpt"), False),
        ("  + PixelOE
  (146.32)", ext_small("ext_recap_gpt_pixeloe"), False),
        ("nano-banana-2
1024 px render", ext_big("ext_recap_nano"), False),
        ("  + downscale
  (87.35)", ext_down("ext_recap_nano"), False),
        ("  + PixelOE
  (150.79)", ext_small("ext_recap_nano_pixeloe"), False),
        ("Ours + ICG
(27.81)", matched("v7r_gft_w1p25_icg1p5"), True),
    ]
    render(ext, "fig_external16", sep_after=(0, 3, 6))


if __name__ == "__main__":
    main()
