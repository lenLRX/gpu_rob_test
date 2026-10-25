# B300 单 SM PCIe host-read 对照实验

本实验比较同一个 SM 上是否存在一个持续读取 mapped host memory 的 warp，对另一个 pointer-chasing 测量 warp 的影响。所有结果均来自 NVIDIA B300 SXM6 AC 的真实运行。

## 结果

| 指标 | 无 host-read warp | 有 host-read warp | 变化 |
|---|---:|---:|---:|
| 平均延迟 | 374.548 ns | 382.431 ns | +7.883 ns（+2.105%） |
| 中位数 | 376 ns | 384 ns | +8 ns |
| P95 | 452 ns | 460 ns | +8 ns |
| P99 | 488 ns | 488.2 ns | 约不变 |
| 最大值 | 548 ns | 532 ns | 正常样本波动 |
| 同期 host-read 请求带宽 | 0 | 1.060 GB/s | — |

结论：两组直方图主体形状相似，但 host-read 组的主体分布略向右移动，平均延迟约增加 2.1%。在本配置下没有观察到新增的超长延迟尾，P99 基本不变。图中纵轴为样本数，横轴为 ns，两组使用相同显示坐标范围。

图表：`pcie_latency_histograms.png` / `pcie_latency_histograms.svg`。

## 实现配置

- 仅启动一个 block、一个 SM；两组实际都运行在 SM 142，其他 SM 不参与。
- baseline 只有一个测量 warp；host 组包含一个测量 warp 和一个 host-read warp。
- pointer chain 为 512 MiB 单循环随机置换，使用 `ld.global.cv.u32`。
- 每次运行从不同的链位置开始，避免后一个 kernel 复用前一个 kernel 留下的 L2 数据。
- 每次正式运行 4096 个样本，每样本为 8 次串行依赖加载的平均值。
- host buffer 为 256 MiB，使用 `cudaHostAllocMapped` 分配并通过 `cudaHostGetDevicePointer` 取得设备地址。
- host warp 的 32 个 lane 按 16 KiB 间隔分散读取；每轮展开 64 组独立读取以增加在途请求。
- host 带宽按实际 32 × 4 字节请求量和 `%globaltimer` 时间计算，是请求的有效字节带宽，不是对 PCIe 物理链路事务字节数的推断。
- 测量先等待约 5 μs，实际为 5024 ns。测量结束后设置 device done 标志，host warp 才退出。
- 正式 host 并发重叠约 12.588 ms；host checksum、请求进度和重叠时间均已验证非零。
- 两组各自进行预测试和正式测量。直方图使用预测试 min–P99 范围内的 512 个线性桶，原始样本和统计均值不裁剪。

## 数据文件

- `pcie_summary.csv`：两次预测试和两次正式运行的汇总。
- `pcie_statistics.csv`：正式样本的均值、中位数、P95、P99 和最大值。
- `pcie_baseline_samples.csv`、`pcie_host_samples.csv`：全部原始样本。
- `pcie_baseline_histogram.csv`、`pcie_host_histogram.csv`：分桶边界和计数。
- `*_histogram_metadata.csv`：预测试范围、桶宽和上下尾排除数量。
- `full_run.log`：B300 实际运行日志。

已校验每次运行 4096 个正延迟样本、每份直方图 512 桶、桶内和上下尾计数之和为 4096，以及 host 流量和并发重叠非零。

## 复现

远程目录：`/root/gpu_rob_test_b300`。

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_103 -lineinfo \
  pcie_contention.cu -o pcie_contention
mkdir -p results/pcie
./pcie_contention --output results/pcie 2>&1 | tee results/pcie/full_run.log
```

本地绘图：

```bash
cd /home/cc/gpu_rob_test
python3 plot_pcie.py results_b300/pcie
```

源码副本：`../source/pcie_contention.cu` 和 `../source/plot_pcie.py`。
