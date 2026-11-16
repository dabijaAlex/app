#!/usr/bin/env python3
"""plot.py - turn the CSV outputs of bin/lsm into PNG graphs (section 6.3).

Usage:  python scripts/plot.py [OUT_DIR]        (default: out)
Writes into OUT_DIR/plots/:
  paths.png              sample paths (needs --dump-paths), strike dashed
  boundary_<name>.png    LSM early-exercise boundary vs time
  lsm_vs_binomial.png    LSM (+-SE) vs binomial reference per option
  timing_breakdown.png   stacked phase times per option
  convergence.png        |LSM - binomial| vs paths, log-log (needs --convergence)
Requires matplotlib and pandas. Missing input files are skipped.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(out / "results.csv") if (out / "results.csv").exists() else None
    strikes = dict(zip(res["name"], res["K"])) if res is not None else {}

    # 1. sample paths
    sp = out / "sample_paths.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        K = res["K"].iloc[0] if res is not None else None
        fig, ax = plt.subplots(figsize=(9, 5))
        for c in [c for c in df.columns if c.startswith("p")]:
            below = K is not None and df[c].iloc[-1] < K
            ax.plot(df["time"], df[c], lw=0.8, color="tab:orange" if below else "tab:blue", alpha=0.8)
        if K is not None:
            ax.axhline(K, ls="--", color="k", lw=1, label=f"strike K={K:g}")
            ax.legend()
        ax.set(xlabel="time (years)", ylabel="price", title="Simulated price paths (orange: ends below K)")
        fig.tight_layout()
        fig.savefig(plots / "paths.png", dpi=120)
        plt.close(fig)

    # 2. exercise boundaries
    for f in sorted(out.glob("boundary_*.csv")):
        name = f.stem.removeprefix("boundary_")
        df = pd.read_csv(f)
        df = df[df["step"] < df["step"].max()]  # last row = expiry, not a decision
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df["time"], df["boundary_price"], "o-", ms=3, lw=1, label="LSM exercise boundary")
        if name in strikes:
            ax.axhline(strikes[name], ls="--", color="k", lw=1, label=f"strike K={strikes[name]:g}")
        ax.set(xlabel="time (years)", ylabel="price", title=f"Early-exercise boundary: {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / f"boundary_{name}.png", dpi=120)
        plt.close(fig)

    # 3. LSM vs binomial
    if res is not None:
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(res)), 5))
        x = range(len(res))
        ax.errorbar(x, res["lsm_price"], yerr=res["lsm_se"], fmt="o", capsize=3, label="LSM ± SE")
        ax.scatter(x, res["binomial"], marker="x", color="k", zorder=3, label="Binomial (American)")
        ax.set_xticks(list(x), res["name"], rotation=60, ha="right", fontsize=8)
        ax.set(ylabel="price", title="LSM vs reference prices")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "lsm_vs_binomial.png", dpi=120)
        plt.close(fig)

    # 4. timing breakdown
    tf = out / "timings.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        cols = ["generate_s", "accumulate_s", "solve_s", "exercise_s", "final_s", "reference_s"]
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(t)), 5))
        bottom = pd.Series([0.0] * len(t))
        for c in cols:
            ax.bar(t["name"], t[c], bottom=bottom, label=c.removesuffix("_s"))
            bottom += t[c]
        ax.set(ylabel="seconds", title="Phase time per option (median over repeats)")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "timing_breakdown.png", dpi=120)
        plt.close(fig)

    # 5. convergence
    cf = out / "convergence.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(c["paths"], c["abs_error_vs_binomial"], "o-", label="|LSM − binomial|")
        ax.loglog(c["paths"], c["se"], "s--", label="standard error")
        ref = c["se"].iloc[0] * (c["paths"].iloc[0] / c["paths"]) ** 0.5
        ax.loglog(c["paths"], ref, ":", color="k", label="1/√N reference")
        ax.set(xlabel="paths N", ylabel="error", title="LSM convergence")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "convergence.png", dpi=120)
        plt.close(fig)

    print(f"plots written to {plots}/")


if __name__ == "__main__":
    main()
