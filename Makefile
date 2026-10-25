SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

NVCC ?= /usr/local/cuda/bin/nvcc
SM ?= 120
TMA_SLOTS ?= 4
TMA_TILE_BYTES ?= 16384
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_$(SM) -DTMA_SLOTS=$(TMA_SLOTS) -DTMA_TILE_BYTES=$(TMA_TILE_BYTES) -lineinfo

.PHONY: all run clean

all: env_probe benchmark

env_probe: env_probe.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

benchmark: benchmark.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcuda

run: all
	mkdir -p results
	nvidia-smi > results/nvidia_smi.txt
	$(NVCC) --version > results/nvcc_version.txt
	uname -a > results/os.txt
	./env_probe > results/environment_runtime.txt
	./benchmark --mode all --output results 2>&1 | tee results/full_run.log
	python3 plot.py
	python3 document_results.py

clean:
	rm -f benchmark env_probe
