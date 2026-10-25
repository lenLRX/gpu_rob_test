# RTX 5080 全部实验汇总

`summary.csv` 汇总 25 次实际运行，包含空载、预测试、正式满载、带宽扫描和半带宽实验。每行记录 SM 数、平均/最小/最大延迟、同期实测带宽、TMA 字节数、globaltimer 时间窗及并发重叠。逐次原始样本和逐块统计由同名 `*_samples.csv`、`*_blocks.csv` 提供。环境记录见 `environment_runtime.txt`、`nvidia_smi.txt` 和 `full_run.log`。
