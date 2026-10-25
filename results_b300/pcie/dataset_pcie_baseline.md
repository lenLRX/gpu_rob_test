# PCIe 对照：无 host warp 基线

单 SM、仅测量 warp 的 lane 0 做 512 MiB HBM pointer chasing。无 host 背景流量。正式样本 4096 个，每样本 8 次串行加载。

平均 **374.548 ns**，中位数 376 ns，P95 452 ns，P99 488.0 ns，最大 548.0 ns。

文件：`pcie_baseline_samples.csv`、`pcie_baseline_pretest_samples.csv`、`pcie_baseline_histogram.csv`、`pcie_baseline_histogram_metadata.csv`。
