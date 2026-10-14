#include "hardware_prober.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>

// CUDA Kernel: Continuous unrolled FMA instructions to saturate SM FP32 ALUs
__global__ void fma_compute_benchmark_kernel(float* __restrict__ d_out, float a, float b, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float v0 = static_cast<float>(tid);
    float v1 = v0 + 1.0f;
    float v2 = v1 + 1.0f;
    float v3 = v2 + 1.0f;
    float v4 = v3 + 1.0f;
    float v5 = v4 + 1.0f;
    float v6 = v5 + 1.0f;
    float v7 = v6 + 1.0f;

    #pragma unroll 16
    for (int i = 0; i < iterations; ++i) {
        v0 = fmaf(v0, a, b);
        v1 = fmaf(v1, a, b);
        v2 = fmaf(v2, a, b);
        v3 = fmaf(v3, a, b);
        v4 = fmaf(v4, a, b);
        v5 = fmaf(v5, a, b);
        v6 = fmaf(v6, a, b);
        v7 = fmaf(v7, a, b);

        v0 = fmaf(v0, a, b);
        v1 = fmaf(v1, a, b);
        v2 = fmaf(v2, a, b);
        v3 = fmaf(v3, a, b);
        v4 = fmaf(v4, a, b);
        v5 = fmaf(v5, a, b);
        v6 = fmaf(v6, a, b);
        v7 = fmaf(v7, a, b);
    }

    if (tid == 0) {
        *d_out = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
}

// CUDA Kernel: Vectorized 128-bit streaming DRAM load/store (float4)
__global__ void bandwidth_benchmark_kernel(const float4* __restrict__ src, float4* __restrict__ dst, size_t n_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n_elements; i += stride) {
        dst[i] = src[i];
    }
}

static double measure_cuda_peak_tflops(int num_sms) {
    float* d_out = nullptr;
    if (cudaMalloc(&d_out, sizeof(float)) != cudaSuccess) {
        return 0.0;
    }

    int threads = 256;
    int blocks = (num_sms > 0) ? num_sms * 16 : 512;
    int iterations = 10000;
    // Each thread performs 16 FMAs per iteration (32 FLOPs per iteration)
    double total_flops = static_cast<double>(blocks) * threads * iterations * 16.0 * 2.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        fma_compute_benchmark_kernel<<<blocks, threads>>>(d_out, 1.0001f, 0.9999f, 500);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    int benchmark_runs = 10;
    for (int r = 0; r < benchmark_runs; ++r) {
        fma_compute_benchmark_kernel<<<blocks, threads>>>(d_out, 1.0001f, 0.9999f, iterations);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);

    if (elapsed_ms <= 0.0f) return 0.0;
    double elapsed_sec = (elapsed_ms / 1000.0) / benchmark_runs;
    return (total_flops / elapsed_sec) * 1e-12; // TFLOPS
}

static double measure_cuda_peak_bandwidth(size_t buffer_size_bytes = 256 * 1024 * 1024) {
    size_t n_elements = buffer_size_bytes / sizeof(float4);
    float4* d_src = nullptr;
    float4* d_dst = nullptr;

    if (cudaMalloc(&d_src, buffer_size_bytes) != cudaSuccess ||
        cudaMalloc(&d_dst, buffer_size_bytes) != cudaSuccess) {
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        return 0.0;
    }
    cudaMemset(d_src, 1, buffer_size_bytes);

    int threads = 256;
    int blocks = 512;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        bandwidth_benchmark_kernel<<<blocks, threads>>>(d_src, d_dst, n_elements);
    }
    cudaDeviceSynchronize();

    // Benchmark
    int benchmark_runs = 10;
    cudaEventRecord(start);
    for (int r = 0; r < benchmark_runs; ++r) {
        bandwidth_benchmark_kernel<<<blocks, threads>>>(d_src, d_dst, n_elements);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);

    if (elapsed_ms <= 0.0f) return 0.0;
    double elapsed_sec = (elapsed_ms / 1000.0) / benchmark_runs;
    double total_bytes = 2.0 * static_cast<double>(buffer_size_bytes); // Read + Write
    return (total_bytes / elapsed_sec) * 1e-9; // GB/s
}
#endif

// Fallback database of published specifications for common deep learning GPUs
HardwareProbeResult lookup_gpu_database_specs(const std::string& gpu_name) {
    std::string lower_name = gpu_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    HardwareProbeResult res;
    res.device_name = gpu_name.empty() ? "NVIDIA Generic GPU" : gpu_name;
    res.precision = "FP32";
    res.is_cuda_calibrated = false;

    if (lower_name.find("h100") != std::string::npos) {
        res.peak_tflops = 67.0;
        res.peak_bandwidth_gb_s = 3350.0;
        res.status_message = "Matched NVIDIA H100 SXM5 / PCIe specifications";
    } else if (lower_name.find("a100") != std::string::npos) {
        res.peak_tflops = 19.5;
        res.peak_bandwidth_gb_s = 1555.0;
        res.status_message = "Matched NVIDIA A100 SXM4 specifications";
    } else if (lower_name.find("v100") != std::string::npos) {
        res.peak_tflops = 15.7;
        res.peak_bandwidth_gb_s = 900.0;
        res.status_message = "Matched NVIDIA V100 SXM2 specifications";
    } else if (lower_name.find("t4") != std::string::npos) {
        res.peak_tflops = 8.1;
        res.peak_bandwidth_gb_s = 320.0;
        res.status_message = "Matched NVIDIA Tesla T4 specifications (Kaggle/Colab)";
    } else if (lower_name.find("4090") != std::string::npos) {
        res.peak_tflops = 82.6;
        res.peak_bandwidth_gb_s = 1008.0;
        res.status_message = "Matched NVIDIA GeForce RTX 4090 specifications";
    } else if (lower_name.find("3090") != std::string::npos) {
        res.peak_tflops = 35.6;
        res.peak_bandwidth_gb_s = 936.0;
        res.status_message = "Matched NVIDIA GeForce RTX 3090 specifications";
    } else if (lower_name.find("3080") != std::string::npos) {
        res.peak_tflops = 29.8;
        res.peak_bandwidth_gb_s = 760.0;
        res.status_message = "Matched NVIDIA GeForce RTX 3080 specifications";
    } else if (lower_name.find("a10g") != std::string::npos || lower_name.find("a10") != std::string::npos) {
        res.peak_tflops = 31.2;
        res.peak_bandwidth_gb_s = 600.0;
        res.status_message = "Matched NVIDIA A10G Tensor Core specifications";
    } else if (lower_name.find("l4") != std::string::npos) {
        res.peak_tflops = 30.3;
        res.peak_bandwidth_gb_s = 300.0;
        res.status_message = "Matched NVIDIA L4 specifications";
    } else {
        // Institutional baseline estimation (Tesla T4 / mid-tier accelerator equivalent)
        res.peak_tflops = 10.0;
        res.peak_bandwidth_gb_s = 400.0;
        res.status_message = "Default institutional baseline calibration applied";
    }

    res.ridge_point = (res.peak_tflops * 1000.0) / res.peak_bandwidth_gb_s;
    return res;
}

HardwareProbeResult probe_gpu_hardware(int device_id) {
#ifdef __CUDACC__
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        if (device_id < 0 || device_id >= device_count) {
            device_id = 0;
        }
        cudaSetDevice(device_id);

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            double empirical_tflops = measure_cuda_peak_tflops(prop.multiProcessorCount);
            double empirical_gb_s = measure_cuda_peak_bandwidth();

            if (empirical_tflops > 0.0 && empirical_gb_s > 0.0) {
                HardwareProbeResult res;
                res.device_name = prop.name;
                res.peak_tflops = empirical_tflops;
                res.peak_bandwidth_gb_s = empirical_gb_s;
                res.ridge_point = (res.peak_tflops * 1000.0) / res.peak_bandwidth_gb_s;
                res.precision = "FP32";
                res.is_cuda_calibrated = true;
                res.status_message = "Empirically calibrated via CUDA unrolled FMA and vectorized 128-bit DRAM streaming kernels";
                return res;
            } else {
                // Microbenchmark launch failed, fall back to specs database for this device name
                HardwareProbeResult fallback = lookup_gpu_database_specs(prop.name);
                fallback.status_message = "CUDA active, but microbenchmarks timed out. Used whitepaper specs database.";
                return fallback;
            }
        }
    }
#endif

    // Fallback path when compiled without NVCC or when no active CUDA device is detected
    return lookup_gpu_database_specs("Generic NVIDIA GPU");
}
