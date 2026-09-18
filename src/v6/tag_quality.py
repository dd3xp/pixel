"""Write caption CSVs with a craft-quality tag prepended, for quality-conditioned training.

09-18: the corpus is community art of very uneven craft (a vision model rates 25% of it 1-2 out of 5), and a model
fitted to the mixture produces the mixture's average. Rather than throwing the low-rated sprites away, we tag every
caption with its rating and let the text encoder carry it, the way aesthetic tags are used when fine-tuning large
text-to-image models; at sampling time we always ask for the top tag. This costs no architecture change.

Tags (prepended to the caption): 5 -> "crisp detailed sprite, ", 4 -> "clean sprite, ", 3 -> "", 1-2 -> "rough sprite, ".

Usage: python src/v6/tag_quality.py --csv data/oga_captions_recap.csv --scores runs/quality_oga.jsonl \
           --out data/oga_captions_q.csv
"""
import argparse
import csv
import json
from pathlib import Path

TAG = {5: "crisp detailed sprite, ", 4: "clean sprite, ", 3: "", 2: "rough sprite, ", 1: "rough sprite, "}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = {}
    for line in open(a.scores, encoding="utf-8"):
        r = json.loads(line)
        sc[r["path"]] = int(r["score"])
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    n_tagged = 0
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        for r in rows:
            s = sc.get(Path(r["path"]).name)
            tag = TAG.get(s, "") if s else ""
            n_tagged += bool(tag)
            w.writerow([r["path"], tag + r["text"]])
    print(f"{a.out}: {len(rows)} rows, {n_tagged} tagged, {len(rows) - sum(1 for r in rows if Path(r['path']).name in sc)} unscored",
          flush=True)


if __name__ == "__main__":
    main()
