# 31 个 host warp 下的大样本 HBM 数据

单 SM、31 个 host PCIe warp，仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing。预测试和正式测量各 **65536 个样本**，每样本 8 次串行加载，正式共 524288 次。

正式平均 **3387.470 ns**，中位数 3356 ns，P95 3720 ns，P99 4000 ns，最大 15440.0 ns。host 请求带宽 1.986 GB/s。

文件：`hbm31_samples.csv`、`hbm31_pretest_samples.csv`、`hbm31_summary.csv`、`hbm31_statistics.csv`、`hbm31_histogram.csv`、`hbm31_histogram_metadata.csv`。
