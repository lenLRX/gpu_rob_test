# NVIDIA B300 同 SM 满载数据集

实验结构：所有 SM 执行 TMA，测量 SM 额外运行一个测量 warp。预测试和正式测量各 4096 个样本，实际 TMA 进度与测量窗口重叠。

正式结果：平均 **983.909 ns**，同期带宽 **7441.360 GB/s**，重叠 32.343 ms。预测试平均 977.438 ns。

直方图使用预测试 min–P99 的 512 个线性桶，范围 512.0–1416.0 ns，桶宽 1.768 ns；桶内 4057.0 个，低于下限 2.0 个，高于 P99 37.0 个。裁剪不影响原始数据和平均值。

文件：`concurrent_samples.csv`、`concurrent_pretest_samples.csv`、`concurrent_blocks.csv`、`concurrent_histogram.csv`、`concurrent_histogram_metadata.csv`。
