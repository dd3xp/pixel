"""Bake the study package into one self-contained HTML page.

The raters are people outside the project opening a link, so the page cannot depend on anything being
fetched or granted: the sprites go in as data URIs and the results come back as a copyable block. A
db write is attempted too, but only as a convenience for viewers who happen to have write access.

Usage: python src/v6/bake_study_page.py --dir runs/study_web --out runs/study_web/index.html
"""
import argparse
import base64
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/study_web")
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    tpl = Path(a.template) if a.template else d / "template.html"
    out = Path(a.out) if a.out else d / "index.html"

    study = json.loads((d / "study.json").read_text(encoding="utf-8"))
    used = set()
    for t in study["judge"] + study["screen"]:
        used.add(t["left_img"])
        used.add(t["right_img"])

    imgs = {}
    for rel in sorted(used):
        raw = (d / rel).read_bytes()
        imgs[rel] = "data:image/png;base64," + base64.b64encode(raw).decode()

    def swap(t):
        t = dict(t)
        t["left_img"] = imgs[t["left_img"]]
        t["right_img"] = imgs[t["right_img"]]
        return t

    data = {
        "questions": study["questions"],
        "judge": [swap(t) for t in study["judge"]],
        "screen": [swap(t) for t in study["screen"]],
    }
    blob = "const STUDY = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";"
    html = tpl.read_text(encoding="utf-8").replace("/*__STUDY_DATA__*/", blob)
    out.write_text(html, encoding="utf-8", newline="")
    mb = len(html.encode("utf-8")) / 1e6
    print(f"{len(imgs)} images inlined, {len(data['judge'])} judgement + {len(data['screen'])} screening "
          f"trials -> {out} ({mb:.2f} MB)")
    if mb > 15:
        raise SystemExit("page exceeds the 16 MB artifact limit")


if __name__ == "__main__":
    main()
