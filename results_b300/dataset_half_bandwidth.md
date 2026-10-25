# NVIDIA B300 半带宽数据集

采用同 SM 并发结构，从扫描点中自动选择实测带宽最接近满载 50% 的配置。选中 **36 个 TMA SM**，正式平均延迟 **435.188 ns**，同期带宽 **3826.358 GB/s**，为同轮满载的 **51.42%**。

预测试和正式测量各 4096 个样本。直方图为预测试 min–P99 的 512 个线性桶，范围 312.0–544.0 ns，桶宽 0.455 ns。

文件：`half_bandwidth_samples.csv`、`half_bandwidth_pretest_samples.csv`、`half_bandwidth_blocks.csv`、`half_bandwidth_histogram.csv`、`half_bandwidth_histogram_metadata.csv`。
