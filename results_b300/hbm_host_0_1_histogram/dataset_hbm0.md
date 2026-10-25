# HBM lane 0：0 个 host warp 数据集

单 SM 中仅测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing；无 host 背景流量。预测试和正式测量各 65536 个样本，每样本 8 次串行加载。

正式平均 **382.304 ns**，中位数 376 ns，P95 456 ns，P99 488 ns，最大 10888.0 ns。host 请求带宽 0.000 GB/s。

直方图范围 184.0–484.0 ns，512 个线性桶，桶宽 0.588 ns。文件：`hbm0_samples.csv`、`hbm0_pretest_samples.csv`、`hbm0_histogram.csv`、`hbm0_histogram_metadata.csv`。
