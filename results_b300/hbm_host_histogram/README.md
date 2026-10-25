# B300：31 个 host PCIe warp 下的 HBM 延迟分布

本实验沿用单 lane 测量结构：一个 SM 中仅测量 warp 的 lane 0 执行 HBM pointer chasing，同时运行 31 个 host-read warp。预测试和正式测量均采集 65536 个样本。

## 正式结果

| 指标 | 数值 |
|---|---:|
| 样本数 | 65536 |
| 每样本串行加载数 | 8 |
| 总正式 HBM 加载数 | 524288 |
| 平均延迟 | 3387.470 ns |
| 中位数 | 3356 ns |
| P95 | 3720 ns |
| P99 | 4000 ns |
| 最大值 | 15440 ns |
| 同期 host 请求带宽 | 1.986 GB/s |
| host/HBM 连续重叠 | 约 1.968 秒 |

图表：`hbm31_latency_histogram.png` / `.svg`。

## 配置与直方图

- HBM 使用完整 512 MiB 单循环随机链和 `ld.global.cv.u32`，仅 lane 0 测量。
- host 使用 256 MiB mapped pinned memory；31 个 host warp 在测量结束前持续运行。
- 预测试与正式测量使用不同链起点，均实际采集 65536 个样本。
- 所有时间使用 `%globaltimer`，分辨率 32 ns。
- 直方图使用预测试 min–P99 范围内的 512 个线性桶。
- 范围为 2836–4004 ns，桶宽 2.283 ns。
- 桶内显示 64897 个正式样本；另有 2 个低于预测试最小值、637 个高于预测试 P99。
- 裁剪仅影响显示，原始样本和统计值保留完整分布。

## 文件与校验

- `hbm31_samples.csv`：65536 个正式原始样本。
- `hbm31_pretest_samples.csv`：65536 个预测试原始样本。
- `hbm31_histogram.csv`、`hbm31_histogram_metadata.csv`：桶边界、计数和范围信息。
- `hbm31_summary.csv`：预测试和正式运行汇总。
- `hbm31_statistics.csv`：均值、中位数、P95、P99 和最大值。
- `full_run.log`：B300 实际运行日志。

已验证样本数、正延迟、512 桶、桶内和上下尾计数之和为 65536，以及两次运行均超过 1.9 秒的真实 host 并发。

## 复现

```bash
cd /root/gpu_rob_test_b300
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_103 -lineinfo \
  hbm_host_histogram.cu -o hbm_host_histogram
mkdir -p results/hbm_host_histogram
./hbm_host_histogram results/hbm_host_histogram \
  2>&1 | tee results/hbm_host_histogram/full_run.log
```

本地绘图：

```bash
cd /home/cc/gpu_rob_test
python3 plot_hbm_host_histogram.py results_b300/hbm_host_histogram
```

源码副本：`../source/hbm_host_histogram.cu` 和 `../source/plot_hbm_host_histogram.py`。
