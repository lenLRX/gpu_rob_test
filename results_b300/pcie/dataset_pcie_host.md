# PCIe 对照：单 host warp 对照

单 SM、仅测量 warp 的 lane 0 做 512 MiB HBM pointer chasing。同期 host 请求带宽 1.060 GB/s，重叠 12.588 ms。正式样本 4096 个，每样本 8 次串行加载。

平均 **382.431 ns**，中位数 384 ns，P95 460 ns，P99 488.2 ns，最大 532.0 ns。

文件：`pcie_host_samples.csv`、`pcie_host_pretest_samples.csv`、`pcie_host_histogram.csv`、`pcie_host_histogram_metadata.csv`。
