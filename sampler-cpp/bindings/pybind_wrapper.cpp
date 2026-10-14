#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "main.h"

namespace py = pybind11;

static py::dict probe_hardware_py(int device_id = 0) {
    HardwareProbeResult res = probe_hardware(device_id);
    py::dict d;
    d["device_name"] = res.device_name;
    d["peak_tflops"] = res.peak_tflops;
    d["peak_bandwidth_gb_s"] = res.peak_bandwidth_gb_s;
    d["ridge_point"] = res.ridge_point;
    d["precision"] = res.precision;
    d["is_cuda_calibrated"] = res.is_cuda_calibrated;
    d["status_message"] = res.status_message;
    return d;
}

static py::dict lookup_hardware_specs_py(const std::string& gpu_name) {
    HardwareProbeResult res = lookup_hardware_specs(gpu_name);
    py::dict d;
    d["device_name"] = res.device_name;
    d["peak_tflops"] = res.peak_tflops;
    d["peak_bandwidth_gb_s"] = res.peak_bandwidth_gb_s;
    d["ridge_point"] = res.ridge_point;
    d["precision"] = res.precision;
    d["is_cuda_calibrated"] = res.is_cuda_calibrated;
    d["status_message"] = res.status_message;
    return d;
}

PYBIND11_MODULE(llm_profiler_sampler, m) {
    m.doc() = "C++ high-frequency system resources sampler & GPU hardware prober for LLM Pipeline Profiler";
    m.def("start_sampler", &start_sampler, "Start background C++ resource sampler thread",
          py::arg("socket_path"), py::arg("interval_ms"));
    m.def("stop_sampler", &stop_sampler, "Stop background C++ resource sampler thread");
    m.def("probe_hardware", &probe_hardware_py, "Empirically probe GPU compute and bandwidth ceilings via microbenchmarks",
          py::arg("device_id") = 0);
    m.def("lookup_hardware_specs", &lookup_hardware_specs_py, "Look up theoretical peak compute and bandwidth specifications",
          py::arg("gpu_name") = "");
}
