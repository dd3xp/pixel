"""Per-resolution TV-vs-FD plot (paper_outline gap #7): x = TV of the pure weak reference relative to the strong model's
own TV at that resolution, y = guided FD relative to bare CFG w=4.  Sources: dmech (log 09-07 14:20, 16 px) and dmech2
(log 09-08 02:00, 20/24 px belief statistics).  Run from repo root: python paper_assets/make_tv_fd_r.py
-> paper_assets/fig_tv_vs_fd_r.png.  Only references whose belief TV was measured are plotted.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# R: (strong TV, bare FD, {label: (ref TV, guided FD)})
DATA = {
    16: (32.0, 21.98, {"bucket:12": (21.6, 8.59), "snapshot 10k": (27.3, 8.98), "bucket:24": (31.9, 17.27),
                       "bucket:64": (40.9, 19.34), "unconditional*": (27.3, 21.98)}),
    20: (31.0, 45.92, {"bucket:12": (16.9, 35.85), "bucket:16": (21.8, 32.89), "bucket:24": (28.5, 46.62)}),
    24: (30.3, 79.80, {"bucket:16": (18.2, 60.41), "bucket:32": (31.3, 101.67)}),
}


def main():
    fig, ax = plt.subplots(figsize=(6, 4))
    for (R, (tv_s, fd_b, pts)), c in zip(DATA.items(), ["C0", "C1", "C2"]):
        for name, (tv, fd) in pts.items():
            x, y = tv / tv_s, fd / fd_b
            ax.scatter(x, y, c=c, s=55, marker="o" if "*" not in name else "s", label=f"{R} px" if name == list(pts)[0] else None)
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 3 if R != 24 else -10), fontsize=7, color=c)
    ax.axvline(1.0, ls="--", c="gray", lw=1)
    ax.axhline(1.0, ls=":", c="gray", lw=1)
    ax.text(1.01, 0.45, "reference TV = strong TV", fontsize=7, color="gray", rotation=90, va="bottom")
    ax.text(0.52, 1.02, "no gain over bare CFG", fontsize=7, color="gray")
    ax.set_xlabel("TV(pure weak reference) / TV(strong model)  at the same resolution")
    ax.set_ylabel("FD after guidance / FD of bare CFG w=4")
    ax.set_title("Lower-contrast, structure-aligned beliefs help; higher-contrast ones do not (16/20/24 px)\n"
                 "* unconditional: low TV but not a same-caption belief -> no gain", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0.48, 1.35)
    fig.tight_layout()
    fig.savefig("paper_assets/fig_tv_vs_fd_r.png", dpi=200)
    print("wrote paper_assets/fig_tv_vs_fd_r.png")


if __name__ == "__main__":
    main()
