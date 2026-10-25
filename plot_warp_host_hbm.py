#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_b300/warp_host_hbm")
summary = pd.read_csv(RESULTS / "warp_summary.csv")
statistics = []
for name in ["warp_host", "warp_mixed"]:
    values = pd.read_csv(RESULTS / f"{name}_samples.csv").latency_ns
    row = summary[summary.experiment == name].iloc[0]
    statistics.append({
        "experiment": name,
        "measured_lanes": int(row.measured_lanes),
        "average_ns": values.mean(),
        "median_ns": values.median(),
        "p95_ns": values.quantile(0.95),
        "p99_ns": values.quantile(0.99),
        "max_ns": values.max(),
    })
pd.DataFrame(statistics).to_csv(RESULTS / "warp_statistics.csv", index=False)

histograms = {
    name: pd.read_csv(RESULTS / f"{name}_histogram.csv")
    for name in ["warp_host", "warp_mixed"]
}
x_min = min(h.lower_ns.min() for h in histograms.values())
x_max = max(h.upper_ns.max() for h in histograms.values())
y_max = max(h["count"].max() for h in histograms.values()) * 1.1
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
titles = [
    "One warp: all 32 lanes chase host memory",
    "One warp: lanes 0-15 HBM, lanes 16-31 host\n(only lanes 0-15 shown)",
]
for ax, name, title in zip(axes, ["warp_host", "warp_mixed"], titles):
    histogram = histograms[name]
    centers = (histogram.lower_ns + histogram.upper_ns) / 2
    widths = histogram.upper_ns - histogram.lower_ns
    ax.bar(centers, histogram["count"], width=widths, edgecolor="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Latency per dependent load (ns)")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("Single-warp host/HBM latency histograms (512 linear bins, pretest min-P99)")
fig.tight_layout()
fig.savefig(RESULTS / "warp_host_hbm_histograms.png", dpi=180, bbox_inches="tight")
fig.savefig(RESULTS / "warp_host_hbm_histograms.svg", bbox_inches="tight")
plt.close(fig)
