# RTX 5080 同 SM 满载数据集

实验结构：所有 SM 执行 TMA，测量 SM 额外运行一个测量 warp。预测试和正式测量各 4096 个样本，实际 TMA 进度与测量窗口重叠。

正式结果：平均 **1805.150 ns**，同期带宽 **870.541 GB/s**，重叠 59.287 ms。预测试平均 2031.401 ns。

直方图使用预测试 min–P99 的 512 个线性桶，范围 736.0–3464.0 ns，桶宽 5.330 ns；桶内 4060.0 个，低于下限 10.0 个，高于 P99 26.0 个。裁剪不影响原始数据和平均值。

文件：`concurrent_samples.csv`、`concurrent_pretest_samples.csv`、`concurrent_blocks.csv`、`concurrent_histogram.csv`、`concurrent_histogram_metadata.csv`。
