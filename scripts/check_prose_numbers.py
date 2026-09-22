"""List numbers that appear in the prose of a draft but in none of its tables.

The 44% figure that had to be removed was true, current for another model, and invisible in every table
of the paper it was argued from. A reader who tries to find it concludes we made it up. This does not
decide anything on its own --- plenty of prose numbers legitimately have no table, like a corpus count
--- it just produces the list to look at, so a number that should be checkable is not missed.

Usage: python scripts/check_prose_numbers.py paper_assets/cvpr27/main.tex
"""
import re
import sys
from pathlib import Path

# numbers that are structure rather than measurement
IGNORE = re.compile(r"^(0|1|2|3|4|5|6|7|8|9|10|12|16|20|24|32|48|64|100|224|299|512|768|1500|3000)$")


def split_tables(tex):
    tables, rest, depth, buf = [], [], 0, []
    for line in tex.split("\n"):
        if "\\begin{tabular}" in line:
            depth += 1
        if depth:
            buf.append(line)
        else:
            rest.append(line)
        if "\\end{tabular}" in line and depth:
            depth -= 1
            if not depth:
                tables.append("\n".join(buf))
                buf = []
    return "\n".join(tables), "\n".join(rest)


def numbers(text):
    text = re.sub(r"%.*", "", text)
    return set(re.findall(r"\d+\.\d+", text))


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_assets/cvpr27/main.tex")
    tex = path.read_text(encoding="utf-8")
    # the preamble is full of numbers that are typography, not measurement
    tex = tex.split("begin{document}", 1)[-1]   # no backslash: avoids an escape in this source
    tbl, prose = split_tables(tex)
    in_tables = numbers(tbl)
    only_prose = sorted(n for n in numbers(prose) - in_tables if not IGNORE.match(n))
    print(f"{path}: {len(in_tables)} numbers in tables, {len(only_prose)} decimals in prose only")
    for n in only_prose:
        ctx = re.search(r"[^.]{0,70}\b" + re.escape(n) + r"\b[^.]{0,40}", prose)
        line = " ".join(ctx.group(0).split()) if ctx else ""
        print(f"  {n:>7}  {line[:110]}")


if __name__ == "__main__":
    main()
