#ifndef HARDWARE_PROBER_CUH
#define HARDWARE_PROBER_CUH

#include <string>

struct HardwareProbeResult {
    std::string device_name;
    double peak_tflops;            // Empirical/calibrated peak FP32 compute ceiling [TFLOPS]
    double peak_bandwidth_gb_s;    // Empirical/calibrated peak DRAM bandwidth ceiling [GB/s]
    double ridge_point;            // Architectural knee point: (P_peak GFLOPS / B_peak GB/s) [FLOPs/Byte]
    std::string precision;         // "FP32", "FP16", "BF16"
    bool is_cuda_calibrated;       // True if measured via active CUDA microbenchmark kernels
    std::string status_message;
};

// Probe hardware capabilities: runs CUDA microbenchmarks if available, else queries fallback database
HardwareProbeResult probe_gpu_hardware(int device_id = 0);

// Lookup empirical whitepaper specs for common GPUs when running without active CUDA runtime
HardwareProbeResult lookup_gpu_database_specs(const std::string& gpu_name);

#endif // HARDWARE_PROBER_CUH
