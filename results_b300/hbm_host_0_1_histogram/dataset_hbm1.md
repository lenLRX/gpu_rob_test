# HBM lane 0：1 个 host warp 数据集

单 SM 中仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing；一个 host warp 连续读取 mapped host memory，不执行 pointer chasing。预测试和正式测量各 65536 个样本，每样本 8 次串行加载。

正式平均 **455.403 ns**，中位数 448 ns，P95 556 ns，P99 596 ns，最大 10732.0 ns。host 请求带宽 0.767 GB/s。

直方图范围 200.0–600.0 ns，512 个线性桶，桶宽 0.783 ns。文件：`hbm1_samples.csv`、`hbm1_pretest_samples.csv`、`hbm1_histogram.csv`、`hbm1_histogram_metadata.csv`。
