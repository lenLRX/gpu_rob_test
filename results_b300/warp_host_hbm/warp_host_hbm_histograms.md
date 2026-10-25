# 单 warp host 与混合访问直方图

对应图：`warp_host_hbm_histograms.png`、`warp_host_hbm_histograms.svg`。左图统计 32 个 host lane，右图仅统计混合 warp 的前 16 个 HBM lane。两组均为预测试 min–P99 的 512 个线性桶，使用相同显示范围。混合组 HBM lane 也表现出接近 host 的延迟，因为统一加载需等待较慢 host lane。

数据说明：`dataset_warp_host.md`、`dataset_warp_mixed.md`。
