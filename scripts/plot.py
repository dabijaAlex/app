#!/usr/bin/env python3
"""plot.py - turn the CSVs written by bin/lsm into two PNGs.

Usage:  python3 scripts/plot.py [OUT_DIR]        (default: out)
Writes into OUT_DIR/:
  paths.png      50 simulated paths, the strike, and the LSM exercise boundary
  boundary.png   exercise boundary (top) and number of paths exercising (bottom)
Requires numpy and matplotlib.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2
INK, MUTED = "#0b0b0b", "#52514e"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    s = np.genfromtxt(out / "summary.csv", delimiter=",", names=True, dtype=None, encoding=None)
    put, K = str(s["type"]) == "put", float(s["K"])
    title = (f"{s['type']}  S0={s['S0']:g} K={K:g} r={s['r']:g} σ={s['sigma']:g} T={s['T']:g}"
             f"   LSM {s['lsm']:.4f} ± {s['se']:.4f}")

    b = np.genfromtxt(out / "boundary.csv", delimiter=",", names=True)
    b = b[:-1]  # last row is expiry, not a decision

    # 1. sample paths + boundary
    p = np.genfromtxt(out / "paths.csv", delimiter=",", skip_header=1)
    t, paths = p[:, 0], p[:, 1:]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    itm = (paths[-1] < K) if put else (paths[-1] > K)
    for i in range(paths.shape[1]):
        ax.plot(t, paths[:, i], lw=0.8, alpha=0.7, color=ORANGE if itm[i] else BLUE)
    ax.plot([], [], color=ORANGE, label="in the money at expiry")
    ax.plot([], [], color=BLUE, label="out of the money at expiry")
    ax.axhline(K, ls="--", color=MUTED, lw=1, label=f"strike K={K:g}")
    ax.plot(b["time"], b["boundary"], color=INK, lw=2, label="LSM exercise boundary")
    ax.set(xlabel="time (years)", ylabel="price", title=title)
    ax.title.set_fontsize(10)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "paths.png", dpi=120)
    plt.close(fig)

    # 2. boundary and exercise counts (two panels, never two y-axes on one)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(b["time"], b["boundary"], "o-", ms=3, lw=2, color=INK, label="exercise boundary")
    a1.axhline(K, ls="--", color=MUTED, lw=1, label=f"strike K={K:g}")
    a1.set(ylabel="price", title=title)
    a1.title.set_fontsize(10)
    a1.legend(frameon=False, fontsize=9)
    w = 0.8 * (b["time"][1] - b["time"][0]) if len(b) > 1 else 0.01
    a2.bar(b["time"], b["itm"], width=w, color="#c9daf2", label="in the money")
    a2.bar(b["time"], b["exercised"], width=w, color=BLUE, label="exercised")
    a2.set(xlabel="time (years)", ylabel="paths")
    a2.legend(frameon=False, fontsize=9)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "boundary.png", dpi=120)
    plt.close(fig)

    print(f"plots written to {out}/paths.png and {out}/boundary.png")


if __name__ == "__main__":
    main()
