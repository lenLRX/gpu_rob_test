#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLanes = 32;
constexpr uint32_t kWarmupLoads = 128;
constexpr uint32_t kLoadsPerSample = 8;
constexpr uint32_t kBins = 512;
constexpr uint64_t kChainBytes = 256ull * 1024 * 1024;

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t status_ = (call);                                                          \
        if (status_ != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); \
        }                                                                                      \
    } while (false)

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ uint64_t globaltimer_after(uint32_t& value) {
    uint64_t time;
    asm volatile("add.u32 %1, %1, 1; mov.u64 %0, %%globaltimer; sub.u32 %1, %1, 1;"
                 : "=l"(time), "+r"(value) :: "memory");
    return time;
}

__device__ __forceinline__ uint32_t load_cv(const uint32_t* address) {
    uint32_t value;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

__global__ void timer_resolution_kernel(uint64_t* result) {
    uint64_t previous = globaltimer();
    uint64_t minimum = ~uint64_t{0};
    for (int i = 0; i < 10000; ++i) {
        uint64_t current = globaltimer();
        uint64_t delta = current - previous;
        if (delta != 0 && delta < minimum) minimum = delta;
        previous = current;
    }
    *result = minimum;
}

__global__ void __launch_bounds__(32, 1) warp_kernel(
    const uint32_t* hbm_chain, const uint32_t* host_chain, uint32_t slice_count,
    uint32_t start, uint32_t samples, uint64_t* latencies, uint32_t mixed,
    uint32_t* smid_result, uint64_t* final_indices) {
    const uint32_t lane = threadIdx.x;
    const uint32_t* base = mixed
        ? (lane < 16 ? hbm_chain + lane * slice_count
                     : host_chain + (lane - 16) * slice_count)
        : host_chain + lane * slice_count;
    uint32_t index = (start + lane * 65537u) % slice_count;
    for (uint32_t i = 0; i < kWarmupLoads; ++i) index = load_cv(base + index);
    for (uint32_t sample = 0; sample < samples; ++sample) {
        uint64_t before = globaltimer();
        #pragma unroll
        for (uint32_t load = 0; load < kLoadsPerSample; ++load) {
            index = load_cv(base + index);
        }
        uint64_t after = globaltimer_after(index);
        latencies[sample * kLanes + lane] = (after - before) / kLoadsPerSample;
    }
    final_indices[lane] = index;
    if (lane == 0) {
        uint32_t value;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
        *smid_result = value;
    }
}

struct RunResult {
    std::string name;
    uint32_t mixed = 0;
    uint32_t measured_lanes = 0;
    uint32_t samples_per_lane = 0;
    double average_ns = 0;
    uint64_t minimum_ns = 0;
    uint64_t maximum_ns = 0;
    uint32_t smid = 0;
    std::vector<uint64_t> latencies;
};

void write_samples(const std::string& path, const RunResult& result) {
    std::ofstream output(path);
    output << "sample,lane,latency_ns\n";
    size_t row = 0;
    for (uint32_t sample = 0; sample < result.samples_per_lane; ++sample) {
        for (uint32_t lane = 0; lane < result.measured_lanes; ++lane) {
            output << sample << ',' << lane << ',' << result.latencies[row++] << '\n';
        }
    }
}

void build_histogram(const std::string& path, const RunResult& formal,
                     const RunResult& pretest) {
    std::vector<uint64_t> sorted = pretest.latencies;
    std::sort(sorted.begin(), sorted.end());
    uint64_t lower = sorted.front();
    size_t index = static_cast<size_t>(std::ceil(0.99 * (sorted.size() - 1)));
    uint64_t p99 = sorted[index];
    uint64_t upper = p99 + 1;
    if (lower == 0 || upper <= lower) throw std::runtime_error("Invalid histogram range");
    double width = static_cast<double>(upper - lower) / kBins;
    std::vector<uint64_t> counts(kBins, 0);
    uint64_t below = 0;
    uint64_t above = 0;
    for (uint64_t latency : formal.latencies) {
        if (latency < lower) ++below;
        else if (latency >= upper) ++above;
        else ++counts[static_cast<uint32_t>((latency - lower) / width)];
    }
    std::ofstream output(path);
    output << "bin,lower_ns,upper_ns,count\n";
    for (uint32_t bin = 0; bin < kBins; ++bin) {
        output << bin << ',' << std::fixed << std::setprecision(6) << lower + bin * width << ','
               << lower + (bin + 1) * width << ',' << counts[bin] << '\n';
    }
    std::ofstream metadata(path.substr(0, path.size() - 4) + "_metadata.csv");
    metadata << "bins,pretest_min_ns,pretest_p99_ns,pretest_max_ns,bin_width_ns,"
                "formal_in_range,formal_below_min,formal_above_p99\n";
    metadata << kBins << ',' << lower << ',' << p99 << ',' << pretest.maximum_ns << ','
             << std::setprecision(17) << width << ',' << formal.latencies.size() - below - above
             << ',' << below << ',' << above << '\n';
}

class Benchmark {
  public:
    explicit Benchmark(uint32_t samples) : samples_(samples) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (!prop.canMapHostMemory) throw std::runtime_error("Mapped host memory unsupported");
        std::cout << "GPU=" << prop.name << " sm_count=" << prop.multiProcessorCount
                  << " l2_bytes=" << prop.l2CacheSize << '\n';
        CUDA_CHECK(cudaMalloc(&hbm_chain_, kChainBytes));
        CUDA_CHECK(cudaHostAlloc(&host_chain_, kChainBytes, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&device_host_chain_, host_chain_, 0));
        CUDA_CHECK(cudaMallocManaged(&latencies_, samples_ * kLanes * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&final_indices_, kLanes * sizeof(uint64_t)));
        CUDA_CHECK(cudaMallocManaged(&smid_, sizeof(uint32_t)));
        initialize_hbm();
        timer_resolution_ns_ = timer_resolution();
        std::cout << "hbm_chain_bytes=" << kChainBytes << " host_chain_bytes=" << kChainBytes
                  << " globaltimer_resolution_ns=" << timer_resolution_ns_ << '\n';
    }

    ~Benchmark() {
        cudaFree(hbm_chain_);
        cudaFreeHost(host_chain_);
        cudaFree(latencies_);
        cudaFree(final_indices_);
        cudaFree(smid_);
    }

    RunResult run(const std::string& name, bool mixed) {
        uint32_t slices = mixed ? 16 : 32;
        uint32_t slice_count = kChainBytes / sizeof(uint32_t) / slices;
        initialize_host(slices);
        uint32_t start = (run_serial_++ * 1048576u) % slice_count;
        *smid_ = ~0u;
        CUDA_CHECK(cudaDeviceSynchronize());
        warp_kernel<<<1, 32>>>(hbm_chain_, device_host_chain_, slice_count, start, samples_,
                               latencies_, mixed ? 1 : 0, smid_, final_indices_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        RunResult result;
        result.name = name;
        result.mixed = mixed;
        result.measured_lanes = mixed ? 16 : 32;
        result.samples_per_lane = samples_;
        result.smid = *smid_;
        for (uint32_t sample = 0; sample < samples_; ++sample) {
            for (uint32_t lane = 0; lane < result.measured_lanes; ++lane) {
                result.latencies.push_back(latencies_[sample * kLanes + lane]);
            }
        }
        auto extrema = std::minmax_element(result.latencies.begin(), result.latencies.end());
        result.minimum_ns = *extrema.first;
        result.maximum_ns = *extrema.second;
        result.average_ns = static_cast<double>(std::accumulate(
            result.latencies.begin(), result.latencies.end(), uint64_t{0})) /
                            result.latencies.size();
        std::cout << name << " measured_lanes=" << result.measured_lanes
                  << " avg_ns=" << result.average_ns << " min_ns=" << result.minimum_ns
                  << " max_ns=" << result.maximum_ns << " smid=" << result.smid << '\n';
        write_samples(output_ + "/" + name + "_samples.csv", result);
        return result;
    }

    void set_output(const std::string& output) { output_ = output; }
    uint64_t timer_resolution_ns() const { return timer_resolution_ns_; }

  private:
    std::vector<uint32_t> make_slice(uint32_t count, uint32_t seed) {
        std::vector<uint32_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 random(seed);
        std::shuffle(order.begin(), order.end(), random);
        std::vector<uint32_t> chain(count);
        for (uint32_t i = 0; i < count; ++i) chain[order[i]] = order[(i + 1) % count];
        return chain;
    }

    void initialize_hbm() {
        uint32_t count = kChainBytes / sizeof(uint32_t) / 16;
        uint32_t* destination = static_cast<uint32_t*>(host_chain_);
        for (uint32_t slice = 0; slice < 16; ++slice) {
            std::vector<uint32_t> chain = make_slice(count, 0xb30000 + slice);
            std::memcpy(destination + slice * count, chain.data(), count * sizeof(uint32_t));
        }
        CUDA_CHECK(cudaMemcpy(hbm_chain_, destination, kChainBytes, cudaMemcpyHostToDevice));
    }

    void initialize_host(uint32_t slices) {
        uint32_t count = kChainBytes / sizeof(uint32_t) / slices;
        uint32_t* destination = static_cast<uint32_t*>(host_chain_);
        for (uint32_t slice = 0; slice < slices; ++slice) {
            std::vector<uint32_t> chain = make_slice(count, 0xb30100 + slices * 32 + slice);
            std::memcpy(destination + slice * count, chain.data(), count * sizeof(uint32_t));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    uint64_t timer_resolution() {
        uint64_t* value;
        CUDA_CHECK(cudaMallocManaged(&value, sizeof(uint64_t)));
        timer_resolution_kernel<<<1, 1>>>(value);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t result = *value;
        cudaFree(value);
        return result;
    }

    uint32_t samples_;
    uint32_t run_serial_ = 0;
    uint32_t* hbm_chain_ = nullptr;
    void* host_chain_ = nullptr;
    uint32_t* device_host_chain_ = nullptr;
    uint64_t* latencies_ = nullptr;
    uint64_t* final_indices_ = nullptr;
    uint32_t* smid_ = nullptr;
    uint64_t timer_resolution_ns_ = 0;
    std::string output_ = "results";
};

void write_summary(const std::string& path, const std::vector<RunResult>& results,
                   uint64_t timer_resolution) {
    std::ofstream output(path);
    output << "experiment,mixed,measured_lanes,samples_per_lane,total_samples,average_ns,"
              "min_ns,max_ns,smid,globaltimer_resolution_ns\n";
    for (const auto& result : results) {
        output << result.name << ',' << result.mixed << ',' << result.measured_lanes << ','
               << result.samples_per_lane << ',' << result.latencies.size() << ','
               << std::setprecision(17) << result.average_ns << ',' << result.minimum_ns << ','
               << result.maximum_ns << ',' << result.smid << ',' << timer_resolution << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        uint32_t samples = 4096;
        std::string output = "results";
        for (int i = 1; i < argc; ++i) {
            std::string argument = argv[i];
            if (++i >= argc) throw std::runtime_error("Missing value after " + argument);
            std::string value = argv[i];
            if (argument == "--samples") samples = std::stoul(value);
            else if (argument == "--output") output = value;
            else throw std::runtime_error("Unknown argument: " + argument);
        }
        Benchmark benchmark(samples);
        benchmark.set_output(output);
        RunResult host_pretest = benchmark.run("warp_host_pretest", false);
        RunResult host = benchmark.run("warp_host", false);
        build_histogram(output + "/warp_host_histogram.csv", host, host_pretest);
        RunResult mixed_pretest = benchmark.run("warp_mixed_pretest", true);
        RunResult mixed = benchmark.run("warp_mixed", true);
        build_histogram(output + "/warp_mixed_histogram.csv", mixed, mixed_pretest);
        write_summary(output + "/warp_summary.csv",
                      {host_pretest, host, mixed_pretest, mixed},
                      benchmark.timer_resolution_ns());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
