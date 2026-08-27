"""Colour-ground the captions: read each sprite's actual pixels, name its
dominant colours, and append them to the caption.

Why: the corpus has ~0 occurrences of "silver" and 20 of "grey", so the model
has no way to ground grey/metal words -- "a pixel art iron ingot" collapses to
the gold ingot that dominates the ingot examples.  We fix the grounding at the
caption side rather than by scraping more data.

Two augmentations:
  1. colour names  -- top-2 dominant opaque colours, named via HSV bands.
  2. material words -- ONLY when the caption already says the object is metal
     (ingot/sword/axe/...).  Preference goes to a material word already in the
     caption (filename semantics: "silver_bar"), because pixel-art silver is
     often rendered blue-grey; otherwise fall back to the dominant hue.
     Gating on the noun avoids teaching "iron" to mean "grey stone".

Note: word boundaries are written as lookarounds rather than \\b, because the
escape does not survive shell heredoc transfer to the server.

Usage: python src/v6/color_caption.py <in.csv> <img_dir> <out.csv>
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

W0, W1 = "(?<![a-zA-Z])", "(?![a-zA-Z])"   # word boundaries without backslashes

METAL_NOUNS = re.compile(
    W0 + "(ingot|bar|nugget|sword|blade|dagger|knife|axe|pickaxe|hammer|mace|"
    "spear|halberd|armour|armor|helmet|helm|shield|plate|chain|key|coin|"
    "anvil|tool|scythe|sickle|hoe|shovel|spade|wrench|screw|nail)s?" + W1, re.I)

CAPTION_MATERIAL = [
    (re.compile(W0 + "(silver|steel|iron|tin)" + W1, re.I), "iron steel silver grey metal"),
    (re.compile(W0 + "(gold|golden|brass)" + W1, re.I), "gold golden brass yellow metal"),
    (re.compile(W0 + "(bronze|copper)" + W1, re.I), "bronze copper brown metal"),
]

MATERIAL_WORDS = {
    "grey": "iron steel silver metal",
    "light grey": "iron steel silver metal",
    "dark grey": "iron steel dark metal",
    "white": "silver steel polished metal",
    "gold": "gold golden brass",
    "yellow": "gold golden brass",
    "brown": "bronze copper rusty",
    "orange": "copper bronze",
}


def name_colour(r, g, b):
    """RGB (0-255) -> a coarse colour name. Low saturation goes to the grey
    ramp; otherwise the hue band, modulated by value."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx
    s = 0.0 if mx == 0 else (mx - mn) / mx
    if s < 0.18:
        if v < 0.16:
            return "black"
        if v < 0.40:
            return "dark grey"
        if v < 0.68:
            return "grey"
        if v < 0.88:
            return "light grey"
        return "white"
    d = mx - mn
    if mx == r:
        h = ((g - b) / d) % 6
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    h *= 60
    if h < 15 or h >= 345:
        return "dark red" if v < 0.5 else "red"
    if h < 45:
        return "brown" if v < 0.62 else "orange"
    if h < 70:
        return "gold" if v > 0.55 and s > 0.5 else ("olive" if v < 0.6 else "yellow")
    if h < 160:
        return "dark green" if v < 0.5 else "green"
    if h < 200:
        return "teal" if v < 0.6 else "cyan"
    if h < 255:
        return "navy" if v < 0.5 else "blue"
    if h < 290:
        return "purple"
    return "pink" if v > 0.7 else "magenta"


def dominant(path, k=2):
    im = Image.open(path).convert("RGBA")
    a = np.array(im)
    opaque = a[a[:, :, 3] > 128][:, :3]
    if len(opaque) == 0:
        return []
    names = Counter(name_colour(*px) for px in opaque)
    total = sum(names.values())
    out = [n for n, c in names.most_common(k) if c / total >= 0.12]
    return out or [names.most_common(1)[0][0]]


def main():
    src_csv, img_dir, out_csv = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    rows = list(csv.DictReader(open(src_csv, encoding="utf-8")))
    out, n_mat = [], 0
    for i, r in enumerate(rows):
        p = img_dir / r["path"]
        try:
            cols = dominant(p)
        except Exception:
            out.append((r["path"], r["text"]))
            continue
        text = r["text"].rstrip(", ")
        if cols:
            text = text + ", " + " and ".join(cols)
        if METAL_NOUNS.search(r["text"]):
            extra = None
            for pat, words in CAPTION_MATERIAL:
                if pat.search(r["text"]):
                    extra = words
                    break
            if extra is None and cols and cols[0] in MATERIAL_WORDS:
                extra = MATERIAL_WORDS[cols[0]]
            if extra:
                text = text + ", " + extra
                n_mat += 1
        out.append((r["path"], text))
        if i % 5000 == 0:
            print(f"[{i}/{len(rows)}] {text[:78]}", flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(out)
    print(f"wrote {out_csv}: {len(out)} rows, {n_mat} with material words", flush=True)


if __name__ == "__main__":
    main()
