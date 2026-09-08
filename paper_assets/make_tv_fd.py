"""TV of the pure weak reference vs the FD it produces when used as the guidance reference, 16 px (v7h, seed 0).

Circles: references the model already contains (its own beliefs under another bucket label, an early snapshot, and the
unconditional prediction that CFG uses).  Crosses: the two trained degraded-view branches of Sec. 5.4, whose training
targets have the *lowest* TV of all and which nevertheless guide worst -- low reference TV is necessary, not sufficient.
Numbers: experiment_log 09-07 14:20 (16 px beliefs), 09-07 13:11 (probe_cg), 09-07 19:05/19:35 (probe_cc), 09-08 19:05
(CFG weight curve statistics).  Reproduces paper_assets/fig_tv_vs_fd.png (Figure 3); run from the repo root.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRONG_TV = 32.0          # TV of the CFG w=4 samples
BARE_W4 = 21.98           # bare CFG at the training default
BEST_CFG = 12.24          # bare CFG at its own FD optimum, w=1.5 -- the baseline the paper compares against
# label -> (TV of the pure reference, FD after guidance, dx, dy) ; dx/dy are label offsets in points
BELIEFS = {
    "bucket:12  (w=2)": (21.6, 8.59, 6, -12),
    "snapshot 10k  (w=1.5)": (27.3, 8.98, 6, 6),
    "unconditional = CFG  (w=4)": (27.3, 21.98, 6, 6),
    "bucket:24  (w=2)": (31.9, 17.27, 8, -4),
    "bucket:64  (w=2)": (40.9, 19.34, -30, 8),
}
TRAINED = {
    "block-average branch  (w=1.5)": (14.2, 21.53, 8, -12),
    "block-average branch  (w=2)": (14.2, 35.22, 8, -4),
    "contrast-shrunk branch  (w=1.5)": (15.9, 14.71, 8, -4),
}


def main():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, (tv, fd, dx, dy) in BELIEFS.items():
        ax.scatter(tv, fd, s=70, c="tab:blue", zorder=3)
        ax.annotate(name, (tv, fd), textcoords="offset points", xytext=(dx, dy), fontsize=8)
    for name, (tv, fd, dx, dy) in TRAINED.items():
        ax.scatter(tv, fd, s=90, c="tab:red", marker="x", linewidths=2.5, zorder=3)
        ax.annotate(name, (tv, fd), textcoords="offset points", xytext=(dx, dy), fontsize=8, color="tab:red")
    ax.axvline(STRONG_TV, color="0.5", ls="--", lw=1)
    ax.annotate("strong-model TV = 32.0", (STRONG_TV, 33.5), xytext=(-6, 0), textcoords="offset points",
                fontsize=8, color="0.4", ha="right")
    ax.axhline(BARE_W4, color="0.5", ls=":", lw=1)
    ax.annotate(f"bare CFG $w=4$ = {BARE_W4}", (43.5, BARE_W4), ha="right", xytext=(0, 5), textcoords="offset points",
                fontsize=8, color="0.4")
    ax.axhline(BEST_CFG, color="0.5", ls="-.", lw=1)
    ax.annotate(f"best CFG $w=1.5$ = {BEST_CFG}", (43.5, BEST_CFG), ha="right", xytext=(0, 5), textcoords="offset points",
                fontsize=8, color="0.4")
    ax.set_xlabel(r"TV of the pure weak reference (mean $|\Delta$RGB$|$ over opaque neighbours)")
    ax.set_ylabel(r"FD-DINOv2 @16 px after guidance  ($\downarrow$)")
    ax.set_xlim(12.5, 44)
    ax.set_ylim(5, 38)
    ax.grid(alpha=0.3)
    ax.set_title("Low reference TV is necessary but not sufficient (16 px, v7h, seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig("paper_assets/fig_tv_vs_fd.png", dpi=200)
    print("wrote paper_assets/fig_tv_vs_fd.png")


if __name__ == "__main__":
    main()
