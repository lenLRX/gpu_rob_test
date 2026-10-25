# B300 单 HBM lane 对 0–31 个 PCIe host warp 扫描

本实验在一个 SM 上仅使用测量 warp 的 **lane 0** 执行 HBM pointer chasing，改变同时运行的 host-read warp 数量。测量 warp 的其他 lane 不访问 HBM。0 个 host warp 作为额外基线，正式扫描覆盖 1–31 个。

## 主要结果

| host warp 数 | HBM lane 0 平均延迟 | 同期 host 请求带宽 |
|---:|---:|---:|
| 0 | 377.348 ns | 0 |
| 1 | 455.133 ns | 0.759 GB/s |
| 8 | 1854.166 ns | 1.990 GB/s |
| 16 | 3227.502 ns | 1.981 GB/s |
| 20 | 3389.802 ns | 1.986 GB/s |
| 24 | 3387.051 ns | 1.988 GB/s |
| 31 | 3351.900 ns | 1.993 GB/s |

延迟总体随 host warp 数增加而上升，在约 20 个之后进入 3.35–3.45 μs 平台区，并非严格单调。31 个 host warp 时，单 SM 中共有 32 个 warp、1024 个线程，但 HBM 测量仍仅使用 lane 0。

折线图：`hbm_latency_vs_host_warps.png` / `.svg`。横轴为 host warp 数，纵轴为 HBM lane 0 的平均延迟（ns）。

## 实现配置

- 仅一个 block、一个 SM；所有配置实际运行在 SM 142。
- 仅 lane 0 执行 HBM pointer chasing，使用覆盖完整 **512 MiB** 的单一随机循环链，不使用小于 L2 的子链。
- 加载为 `ld.global.cv.u32`；每配置 4096 个样本，每样本为 8 次串行依赖加载的平均值。
- 每个配置更换链起点，避免直接复用前一次测量的数据。
- 保存 lane 0 的最终索引，维持加载值到结束计时之间的数据依赖。
- host buffer 为 256 MiB `cudaHostAllocMapped` pinned memory。
- 每个 host lane 展开 16 组独立读取；不同 lane 和 warp 使用分散地址，host warp 在测量结束标志设置前持续运行。
- host 带宽按实际 32 × 4 字节请求量及 `%globaltimer` 时间计算，不推断 PCIe 物理事务字节数。
- 每个 host warp 的轮数、checksum 和重叠均验证非零；1–31 配置均有真实 host 流量。
- 所有延迟使用 `%globaltimer`，分辨率 32 ns。

## 数据文件

- `hbm_host_sweep.csv`：32 个配置的 lane 0 平均、最小、最大延迟，以及 host 字节数、带宽、轮数和重叠时间。
- `hbm_lane_averages.csv`：每个配置的 lane 0 平均延迟，共 32 行。
- `full_run.log`：B300 实际运行日志。
- `hbm_latency_vs_host_warps.png` / `.svg`：折线图。

已验证配置数量、host warp 顺序、正延迟、每配置仅有 lane 0 统计，以及 1–31 个 host warp 的流量和并发重叠。

## 复现

```bash
cd /root/gpu_rob_test_b300
/usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_103 -lineinfo \
  hbm_host_sweep.cu -o hbm_host_sweep
mkdir -p results/hbm_host_sweep
./hbm_host_sweep results/hbm_host_sweep \
  2>&1 | tee results/hbm_host_sweep/full_run.log
```

本地绘图：

```bash
cd /home/cc/gpu_rob_test
python3 plot_hbm_host_sweep.py results_b300/hbm_host_sweep
```

源码副本：`../source/hbm_host_sweep.cu` 和 `../source/plot_hbm_host_sweep.py`。
