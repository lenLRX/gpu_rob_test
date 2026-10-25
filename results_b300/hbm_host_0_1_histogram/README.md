# B300：0 与 1 个连续 host-read warp 的 HBM 延迟直方图

两组实验均在一个 SM 上仅使用测量 warp 的 lane 0 执行完整 512 MiB HBM pointer chasing。第一组无 host warp，第二组有一个连续读取 mapped host memory 的 warp；host warp 不执行 pointer chasing。

每组预测试和正式测量各 65536 个样本，每样本为 8 次串行 HBM 加载的平均值。

| host warp 数 | 平均延迟 | 中位数 | P95 | P99 | 最大值 |
|---:|---:|---:|---:|---:|---:|
| 0 | 382.304 ns | 376 ns | 456 ns | 488 ns | 10888 ns |
| 1 | 455.403 ns | 448 ns | 556 ns | 596 ns | 10732 ns |

一个连续 host-read warp 使平均延迟增加约 73.098 ns。host 组同期请求带宽为 0.767 GB/s，连续重叠约 0.240 秒。

图表：`hbm_0_1_latency_histograms.png` / `.svg`，两组使用相同显示坐标和预测试 min–P99 的 512 个线性桶。原始样本及统计不裁剪。

文档索引：`RESULTS_INDEX.md`；数据说明：`dataset_hbm0.md`、`dataset_hbm1.md`；图表说明：`hbm_0_1_latency_histograms.md`。

复现：

```bash
cd /root/gpu_rob_test_b300
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_103 -lineinfo \
  hbm_host_0_1_histogram.cu -o hbm_host_0_1_histogram
mkdir -p results/hbm_host_0_1_histogram
./hbm_host_0_1_histogram results/hbm_host_0_1_histogram \
  2>&1 | tee results/hbm_host_0_1_histogram/full_run.log
```

本地绘图：`python3 plot_hbm_host_0_1_histogram.py results_b300/hbm_host_0_1_histogram`。
