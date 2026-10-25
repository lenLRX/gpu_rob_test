#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_b300/hbm_host_0_1_histogram")
summary = pd.read_csv(RESULTS / "hbm_0_1_summary.csv")
statistics = []
for host_warps in [0, 1]:
    values = pd.read_csv(RESULTS / f"hbm{host_warps}_samples.csv").latency_ns
    row = summary[(summary.host_warps == host_warps)].iloc[-1]
    statistics.append({
        "host_warps": host_warps,
        "samples": len(values),
        "average_ns": values.mean(),
        "median_ns": values.median(),
        "p95_ns": values.quantile(0.95),
        "p99_ns": values.quantile(0.99),
        "max_ns": values.max(),
        "host_bandwidth_GB_per_s": row.host_bandwidth_GB_per_s,
    })
pd.DataFrame(statistics).to_csv(RESULTS / "hbm_0_1_statistics.csv", index=False)

histograms = {
    count: pd.read_csv(RESULTS / f"hbm{count}_histogram.csv") for count in [0, 1]
}
x_min = min(h.lower_ns.min() for h in histograms.values())
x_max = max(h.upper_ns.max() for h in histograms.values())
y_max = max(h["count"].max() for h in histograms.values()) * 1.1
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, count in zip(axes, [0, 1]):
    histogram = histograms[count]
    centers = (histogram.lower_ns + histogram.upper_ns) / 2
    widths = histogram.upper_ns - histogram.lower_ns
    ax.bar(centers, histogram["count"], width=widths, edgecolor="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("HBM lane-0 latency per dependent load (ns)")
    ax.set_ylabel("Samples")
    ax.set_title(f"{count} host warp" + (" (continuous host reads)" if count else ""))
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("HBM latency with 0 vs. 1 host warp (65536 samples, 512 linear bins)")
fig.tight_layout()
fig.savefig(RESULTS / "hbm_0_1_latency_histograms.png", dpi=180, bbox_inches="tight")
fig.savefig(RESULTS / "hbm_0_1_latency_histograms.svg", bbox_inches="tight")
plt.close(fig)
