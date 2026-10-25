#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_b300/hbm_host_histogram")
values = pd.read_csv(RESULTS / "hbm31_samples.csv").latency_ns
summary = pd.read_csv(RESULTS / "hbm31_summary.csv").iloc[-1]
pd.DataFrame([{
    "samples": len(values),
    "average_ns": values.mean(),
    "median_ns": values.median(),
    "p95_ns": values.quantile(0.95),
    "p99_ns": values.quantile(0.99),
    "max_ns": values.max(),
    "host_bandwidth_GB_per_s": summary.host_bandwidth_GB_per_s,
}]).to_csv(RESULTS / "hbm31_statistics.csv", index=False)

histogram = pd.read_csv(RESULTS / "hbm31_histogram.csv")
centers = (histogram.lower_ns + histogram.upper_ns) / 2
widths = histogram.upper_ns - histogram.lower_ns
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(centers, histogram["count"], width=widths, edgecolor="none")
ax.set_xlabel("HBM lane-0 latency per dependent load (ns)")
ax.set_ylabel("Samples")
ax.set_title(f"31 host PCIe warps, {len(values)} HBM samples\n"
             f"Mean {values.mean():.1f} ns, host read {summary.host_bandwidth_GB_per_s:.3f} GB/s")
ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("HBM latency distribution (512 linear bins, pretest min-P99)")
fig.tight_layout()
fig.savefig(RESULTS / "hbm31_latency_histogram.png", dpi=180, bbox_inches="tight")
fig.savefig(RESULTS / "hbm31_latency_histogram.svg", bbox_inches="tight")
plt.close(fig)
