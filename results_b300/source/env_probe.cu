#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("device_count=%d\n", count);
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop{};
        status = cudaGetDeviceProperties(&prop, device);
        if (status != cudaSuccess) {
            std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(status));
            return 1;
        }
        int runtime = 0;
        int driver = 0;
        cudaRuntimeGetVersion(&runtime);
        cudaDriverGetVersion(&driver);
        std::printf("device=%d\nname=%s\ncompute_capability=%d.%d\nsm_count=%d\n",
                    device, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        std::printf("global_memory_bytes=%zu\nl2_cache_bytes=%d\nshared_memory_per_block_optin=%zu\n",
                    prop.totalGlobalMem, prop.l2CacheSize, prop.sharedMemPerBlockOptin);
        std::printf("cuda_runtime_version=%d.%d\ncuda_driver_version=%d.%d\n",
                    runtime / 1000, runtime % 100 / 10, driver / 1000, driver % 100 / 10);
    }
    return count > 0 ? 0 : 1;
}
